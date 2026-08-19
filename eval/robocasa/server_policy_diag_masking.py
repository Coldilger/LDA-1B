"""One-off diagnostic: does the L1-probe anomaly (oracle/world-model L1
worse than a trivial zero-action baseline, see
eval/bridge/experiment1_ablation/README.md) come from averaging L1 over
the full 138-dim padded action space, most of which is structurally
irrelevant for this embodiment (GR1 here has no legs/head/base -- see the
robosuite controller-removal warnings in any RoboCasa eval log)? A trivial
zero baseline gets those dims "right" for free (real value is truly 0
there); if oracle/world-model's raw output isn't cleanly zero on those same
dims, unmasked L1 would penalize them for noise nobody cares about.

Test: log the raw per-dimension real action across real steps, empirically
determine which dims ever move (active) vs never move (dead/padding), then
report L1 both ways -- unmasked (matching the original probe) and masked to
active-dims-only. If masking closes or reverses the oracle/world-model vs.
zero gap, that's the explanation.
"""

import argparse
import logging
import socket

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework


def add_masking_diag(vla):
    original_predict_action = vla.predict_action
    original_video_gen = vla.video_gen

    state = {"prev_example": None, "prev_action": None, "in_probe": False}
    records = []  # each: dict(real, oracle, worldmodel)

    def probed_predict_action(examples, **kwargs):
        if state["in_probe"]:
            return original_predict_action(examples, **kwargs)
        if not isinstance(examples, list):
            examples = [examples]

        if state["prev_example"] is not None and state["prev_action"] is not None:
            state["in_probe"] = True
            try:
                prev_example = state["prev_example"]
                real_action = state["prev_action"]

                this_frame = np.asarray(examples[0]["image"])[-1]
                oracle_future_imgs = np.array([[this_frame]])
                oracle_out = original_predict_action([prev_example], oracle_future_imgs=oracle_future_imgs)
                oracle_pred = oracle_out["normalized_actions"][0]

                video_gen_out = original_video_gen([prev_example])
                imagined_tokens = torch.from_numpy(video_gen_out["normalized_obs"])
                wm_out = original_predict_action([prev_example], inverse_dynamics_next_obs_tokens=imagined_tokens)
                wm_pred = wm_out["normalized_actions"][0]

                records.append({"real": real_action, "oracle": oracle_pred, "worldmodel": wm_pred})
                logging.info("MASKING_DIAG_SAMPLE n=%d recorded", len(records))
            except Exception:
                logging.exception("Probe side-computation failed (real rollout unaffected)")
            finally:
                state["in_probe"] = False

        result = original_predict_action(examples, **kwargs)
        state["prev_example"] = examples[0]
        state["prev_action"] = result["normalized_actions"][0]
        return result

    def summarize():
        if not records:
            logging.info("MASKING_DIAG: no samples recorded")
            return
        real = np.stack([r["real"] for r in records])  # (N, T, D)
        oracle = np.stack([r["oracle"] for r in records])
        wm = np.stack([r["worldmodel"] for r in records])
        zero = np.zeros_like(real)

        # Active dims: ever move by more than a small threshold across all
        # samples/timesteps of the REAL action (data-driven, no assumed mask).
        real_flat = real.reshape(-1, real.shape[-1])
        dim_range = real_flat.max(axis=0) - real_flat.min(axis=0)
        active = dim_range > 1e-3
        n_active = int(active.sum())
        logging.info("MASKING_DIAG active_dims=%d / %d total", n_active, real.shape[-1])
        logging.info("MASKING_DIAG per-dim range (first 20): %s", np.round(dim_range[:20], 4).tolist())

        def l1(pred, ref, mask=None):
            diff = np.abs(pred - ref)
            if mask is not None:
                diff = diff[..., mask]
            return float(diff.mean())

        for name, pred in [("oracle", oracle), ("worldmodel", wm), ("zero", zero)]:
            unmasked = l1(pred, real)
            masked = l1(pred, real, active) if n_active > 0 else float("nan")
            logging.info("MASKING_DIAG condition=%s unmasked_l1=%.5f active_dims_l1=%.5f", name, unmasked, masked)

        # Chunk hypothesis: `real` here is the full 16-step chunk the default
        # `policy` call predicted, but the client only ever executes
        # n_action_steps=12 of it before replanning -- steps 12-16 were
        # computed but never physically validated. Does restricting the
        # comparison to only the steps that were actually executed (or just
        # the very first, most-real step) change the oracle/world-model vs.
        # zero-baseline gap?
        T = real.shape[1]
        for cutoff in [1, 4, 8, 12, T]:
            c = min(cutoff, T)
            for name, pred in [("oracle", oracle), ("worldmodel", wm), ("zero", zero)]:
                v = l1(pred[:, :c], real[:, :c])
                logging.info("MASKING_DIAG chunk_cutoff=%d condition=%s l1=%.5f", c, name, v)

    inner = probed_predict_action

    def probed_predict_action_periodic(examples, **kwargs):
        result = inner(examples, **kwargs)
        # SIGTERM at job end skips Python `finally` blocks -- log periodically
        # so partial results survive an abrupt kill, same reasoning as
        # server_policy_oracle_probe.py.
        if len(records) and len(records) % 10 == 0:
            summarize()
        return result

    vla.predict_action = probed_predict_action_periodic
    vla._masking_diag_summarize = summarize
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    vla = add_masking_diag(vla)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    try:
        server.serve_forever()
    finally:
        vla._masking_diag_summarize()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
