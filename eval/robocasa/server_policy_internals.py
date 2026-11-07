"""Internal sanity check of LDA-1B's world-model-on path (advisor point 2:
shapes, masks, attention, activations, action validity), the counterpart of
mimic-video's experiment1_ablation/ablation_sanity/ and F1-VLA's
experiment1_ablation/internals/.

The robot is driven by the DEFAULT policy path (exactly as the baseline
closed-loop run), so every condition below is compared on identical states.
When the observation of decision k+1 arrives, decision k's example is re-run
through the action head under several contents of the next-obs slot:

  policy     next_obs_learnable_tokens, policy_embedding (the default path)
  wm_on      this model's own video_gen() tokens, id_embedding (Experiment 1)
  oracle     the real frame of decision k+1, encoded as curr_obs is
             (Experiment 2's hook)
  zero       zeros of the wm_on shape
  noise      Gaussian with each wm_on token's own mean/std
  shuffle    wm_on tokens imagined 60 decisions earlier (another episode:
             an episode has at most 720/12 = 60 decisions)

All conditions use the same torch seed per decision (identical initial action
noise), inside torch.random.fork_rng so the driving policy's RNG stream is
untouched (checked: CUDA RNG state hash before/after is logged).

Recorded per condition: shapes, next-obs token statistics, the share of the
obs_merger output that comes from the next-obs half of its input, per-block
attention from action queries (mass on register / obs / action-stream keys,
normalized entropy), kernel-vs-manual attention error, and the action chunk.
For the first decisions a per-token perturbation test of the next-obs slot is
run (a masked token could not change the action).
"""

import argparse
import hashlib
import logging
import math
import pathlib
import socket

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework

SHUFFLE_LAG = 60
N_PERTURB_DECISIONS = 3


class _AttnProbe:
    """Wraps every block's JointAttention to record attention statistics of
    action queries, recomputed from the q/k the kernel receives."""

    def __init__(self, blocks, n_register):
        self.n_register = n_register
        self.enabled = False
        self.calls = []  # per call: dict(block, mass_reg, mass_obs, mass_act, entropy, kernel_err)
        for i, blk in enumerate(blocks):
            ja = blk.joint_attn
            ja._probe_lens = None
            orig_forward = ja.forward

            def fwd(inputs, *a, _ja=ja, _orig=orig_forward, **kw):
                _ja._probe_lens = [x.shape[1] for x in inputs]
                return _orig(inputs, *a, **kw)

            def attend_hook(mod, args, kwargs, output, _ja=ja, _i=i):
                if self.enabled:
                    q, k, v = args[:3]
                    self._record(_i, _ja._probe_lens, q, k, v, kwargs.get("mask"), output[0])

            ja.forward = fwd
            ja.attend.register_forward_hook(attend_hook, with_kwargs=True)

    def _record(self, block, lens, q, k, v, mask, out):
        n_img, n_act = lens
        logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :].bool(), float("-inf"))
        w = logits.softmax(-1)
        manual = torch.einsum("bhqk,bhkd->bhqd", w, v.float())
        kernel_err = float((manual - out.float()).norm() / (out.float().norm() + 1e-12))
        wa = w[:, :, n_img:, :]  # action-stream queries (state + action tokens)
        ent = -(wa * wa.clamp_min(1e-12).log()).sum(-1) / math.log(w.shape[-1])
        self.calls.append({
            "block": block,
            "mass_reg": float(wa[..., : self.n_register].sum(-1).mean()),
            "mass_obs": float(wa[..., self.n_register:n_img].sum(-1).mean()),
            "mass_act": float(wa[..., n_img:].sum(-1).mean()),
            "entropy": float(ent.mean()),
            "kernel_err": kernel_err,
            "mask_all_true": bool(mask is None or bool(mask.all())),
        })


def _tok_stats(t):
    t = t.float()[0]  # [N, C]
    return {
        "shape": list(t.shape),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "token_norm": float(t.norm(dim=-1).mean()),
        "nan": int(torch.isnan(t).sum()),
    }


def _cos(a, b):
    a, b = a.float()[0], b.float()[0]
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def add_internals(vla, out_path):
    head = vla.action_model
    original_predict_action = vla.predict_action
    original_video_gen = vla.video_gen
    n_register = head.register_tokens.weight.shape[0]
    probe = _AttnProbe(head.model.blocks, n_register)

    # obs_merger: Linear over cat([curr_obs, next_obs], -1); record how much of
    # its output comes from the next-obs half (first call per predict_action).
    merger = {"first": True, "rec": None, "curr_obs": None, "next_obs": None}
    C_next = head.vision_encoder.config.hidden_size

    def merger_hook(mod, inp, out):
        if not merger["first"]:
            return
        merger["first"] = False
        x = inp[0].float()
        W = mod.weight.float()
        c_curr = x.shape[-1] - C_next
        from_curr = x[..., :c_curr] @ W[:, :c_curr].T
        from_next = x[..., c_curr:] @ W[:, c_curr:].T
        merger["curr_obs"] = x[..., :c_curr].detach()
        merger["next_obs"] = x[..., c_curr:].detach()
        merger["rec"] = {
            "input_shape": list(x.shape),
            "curr_channels": c_curr,
            "next_channels": C_next,
            "norm_from_curr": float(from_curr.norm(dim=-1).mean()),
            "norm_from_next": float(from_next.norm(dim=-1).mean()),
            "next_share": float(from_next.norm() / (from_curr.norm() + from_next.norm() + 1e-12)),
        }

    head.obs_merger.register_forward_hook(merger_hook)

    st = {"prev": None, "n": 0}
    pool = []  # wm_on tokens per decision, for the shuffle condition
    records = []
    noise_gen = torch.Generator(device="cpu").manual_seed(0)
    cfg = vla.config.framework.action_model
    future_obs_index = int(getattr(cfg, "future_obs_index", head.future_obs_index))

    def run(example, seed, **kw):
        torch.manual_seed(seed)
        merger["first"] = True
        probe.calls = []
        probe.enabled = True
        try:
            a = original_predict_action([example], **kw)["normalized_actions"][0]
        finally:
            probe.enabled = False
        return a, list(probe.calls), merger["rec"]

    def side(example, next_frame, seed):
        rec = {"decision": st["n"], "conditions": {}, "future_obs_index": future_obs_index}
        torch.manual_seed(seed)
        imagined = torch.from_numpy(original_video_gen([example])["normalized_obs"])
        pool.append(imagined)
        conds = {
            "policy": {},
            "wm_on": {"inverse_dynamics_next_obs_tokens": imagined},
            "oracle": {"oracle_future_imgs": np.array([[next_frame]])},
            "zero": {"inverse_dynamics_next_obs_tokens": torch.zeros_like(imagined)},
            "noise": {"inverse_dynamics_next_obs_tokens": imagined.mean(-1, keepdim=True)
                      + imagined.std(-1, keepdim=True) * torch.randn(imagined.shape, generator=noise_gen)},
        }
        if len(pool) > SHUFFLE_LAG:
            conds["shuffle"] = {"inverse_dynamics_next_obs_tokens": pool[-1 - SHUFFLE_LAG]}
        next_tokens = {}
        for name, kw in conds.items():
            a, calls, mrec = run(example, seed, **kw)
            next_tokens[name] = merger["next_obs"]
            if name == "policy":
                curr_tokens = merger["curr_obs"]
            rec["conditions"][name] = {"action": a, "attn": calls, "merger": mrec,
                                       "next_stats": _tok_stats(merger["next_obs"]),
                                       "action_nan": int(np.isnan(a).sum())}
        rec["curr_stats"] = _tok_stats(curr_tokens)
        # curr_obs folds obs_horizon frames into channels as "(c t)" (frame is
        # the fast index); compare forecasts with the most recent frame only.
        c_next = next_tokens["policy"].shape[-1]
        if curr_tokens.shape[-1] != c_next:
            curr_tokens = curr_tokens.reshape(*curr_tokens.shape[:-1], c_next, -1)[..., -1]
        rec["imagined_shape"] = list(imagined.shape)
        rec["cos_wm_vs_oracle"] = _cos(next_tokens["wm_on"], next_tokens["oracle"])
        rec["cos_wm_vs_curr"] = _cos(next_tokens["wm_on"], curr_tokens)
        rec["cos_oracle_vs_curr"] = _cos(next_tokens["oracle"], curr_tokens)
        rec["rel_l2_wm_vs_oracle"] = float((next_tokens["wm_on"] - next_tokens["oracle"]).norm()
                                           / (next_tokens["oracle"].norm() + 1e-12))

        if st["n"] < N_PERTURB_DECISIONS:
            base, _, _ = run(example, seed, inverse_dynamics_next_obs_tokens=imagined)
            N = imagined.shape[1]
            effects = {}
            for pos in sorted({0, N // 4, N // 2, 3 * N // 4, N - 1}):
                pert = imagined.clone()
                pert[:, pos] = torch.randn(pert[:, pos].shape, generator=noise_gen) * imagined.std() + imagined.mean()
                a, _, _ = run(example, seed, inverse_dynamics_next_obs_tokens=pert)
                effects[pos] = float(np.abs(a - base).max())
            again, _, _ = run(example, seed, inverse_dynamics_next_obs_tokens=imagined)
            rec["perturb_effect_per_pos"] = effects
            rec["perturb_repeat_noise"] = float(np.abs(again - base).max())
        return rec

    def save():
        np.savez_compressed(out_path, records=np.array(records, dtype=object))

    def predict_action(examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        if st["prev"] is not None:
            dev = torch.cuda.current_device()
            h0 = hashlib.md5(torch.cuda.get_rng_state(dev).numpy().tobytes()).hexdigest()
            with torch.random.fork_rng(devices=[dev]):
                try:
                    rec = side(st["prev"], np.asarray(examples[0]["image"])[-1], seed=10_000 + st["n"])
                    records.append(rec)
                except Exception:
                    logging.exception("internals side computation failed (rollout unaffected)")
            h1 = hashlib.md5(torch.cuda.get_rng_state(dev).numpy().tobytes()).hexdigest()
            if records:
                records[-1]["rng_untouched"] = h0 == h1
            st["n"] += 1
            if st["n"] % 5 == 0:
                save()
                logging.info("INTERNALS saved %d records", len(records))
        result = original_predict_action(examples, **kwargs)
        st["prev"] = examples[0]
        return result

    vla.predict_action = predict_action
    vla._internals_save = save
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    for name, node in [("action_model", vla.config.framework.action_model),
                       ("vla_data", vla.config.datasets.vla_data)]:
        try:
            from omegaconf import OmegaConf
            node = OmegaConf.to_container(node, resolve=True)
        except Exception:
            pass
        logging.info("INTERNALS_CONFIG %s: %s", name, node)
    vla = add_internals(vla, args.out)

    local_ip = socket.gethostbyname(socket.gethostname())
    logging.info("Creating server (ip: %s)", local_ip)
    server = WebsocketPolicyServer(policy=vla, host="0.0.0.0", port=args.port,
                                   idle_timeout=args.idle_timeout, metadata={"env": "simpler_env"})
    logging.info("server running ...")
    try:
        server.serve_forever()
    finally:
        vla._internals_save()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
