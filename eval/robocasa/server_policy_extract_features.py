"""Experiment 4 (LDA-1B), step 1: extract frozen vl_embs features live via
RoboCasa, for train_probe.py / positive_control.py / compare_targets.py
(copied verbatim from F1-VLA/mimic-video's own experiment4_probing/ --
those three are fully generic over any {features, targets, current_pose,
episode_ids, horizon, variant} .npz, no model-specific code in them).

Extraction point: `vl_embs` -- the shared Qwen3-VL backbone output feeding
both the policy and the (normally-unused) auxiliary heads
(`forward_dynamics`/`inverse_dynamics`/`video_gen`). Chosen over
video_gen()'s own output because it's a single forward pass (no 4-step
diffusion sampling needed per sample) and is the shared representation
upstream of all task-specific branching -- see
mimic-video/eval/bridge/experiment4_probing/README.md's own note on why
`dynamics_loss`-adjacent candidates were ruled out for LDA.

No static RoboCasa dataset exists on this cluster (same constraint as
Experiments 1/2), so this runs the *normal, unmodified* policy live and
captures vl_embs as a side effect of real rollouts -- mechanically
equivalent to F1/mimic's offline extraction, just sourced from a live
simulation instead of a saved file. Robot behavior is completely
unaffected (a pure capture wrapper, like server_policy_diag_frames.py).

Pose target: state (concat of left_arm/right_arm/left_hand/right_hand/waist,
matching model2robocasa_interface_with_history.py's own input_state
construction), analogous to F1/mimic's 9-D end-effector pose but for this
humanoid's own available proprioception. Probe target is the *delta* K
replan-steps ahead (matching F1/mimic's own reasoning for avoiding the
"barely moves" degeneracy at the absolute-pose target); current pose (t) is
saved alongside for compare_targets.py's decisive control.

One job = one episode (its own episode_id = SLURM_JOB_ID), submitted
multiple times for multiple episodes -- avoids needing in-process episode-
boundary detection, at the cost of one checkpoint-load per episode.
"""

import argparse
import logging
import socket
from collections import deque

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework


def add_feature_extraction(vla, episode_id: str, horizon: int, out_path: str):
    original_forward = vla.qwen_vl_interface.forward
    captured = {}

    def capturing_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        captured["hidden"] = out.hidden_states[-1].detach()
        return out

    vla.qwen_vl_interface.forward = capturing_forward

    original_predict_action = vla.predict_action
    buffer = deque(maxlen=horizon + 1)  # (feature, state) pairs, oldest first
    records = {"features": [], "targets": [], "current_pose": []}

    def extracting_predict_action(examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        captured.clear()
        result = original_predict_action(examples, **kwargs)
        if "hidden" not in captured:
            logging.warning("vl_embs was not captured on this call -- skipping")
            return result

        hidden = captured["hidden"].float()  # (B, L, H)
        feat = torch.cat([hidden.mean(dim=1), hidden.std(dim=1)], dim=-1)[0].cpu().numpy()
        state = np.asarray(examples[0]["state"], dtype=np.float64).reshape(-1)

        buffer.append((feat, state))
        if len(buffer) == buffer.maxlen:
            feat0, state0 = buffer[0]
            _, state_k = buffer[-1]
            records["features"].append(feat0)
            records["targets"].append(state_k - state0)
            records["current_pose"].append(state0)
            n = len(records["features"])
            if n == 1:
                logging.info("EXTRACT feature_dim=%d state_dim=%d", feat0.shape[0], state0.shape[0])
            logging.info("EXTRACT_SAMPLE n=%d", n)
            # SIGTERM at job end skips Python `finally` blocks (the SLURM
            # script always `kill`s the server rather than letting
            # serve_forever() return naturally) -- save periodically so
            # partial results survive an abrupt kill.
            if n % 10 == 0:
                save()

        return result

    vla.predict_action = extracting_predict_action

    def save():
        if not records["features"]:
            logging.info("EXTRACT: no samples recorded")
            return
        feats = np.stack(records["features"]).astype(np.float32)
        targets = np.stack(records["targets"]).astype(np.float32)
        current = np.stack(records["current_pose"]).astype(np.float32)
        ep = np.array([episode_id] * len(feats))
        np.savez_compressed(
            out_path,
            features=feats,
            targets=targets,
            current_pose=current,
            episode_ids=ep,
            horizon=horizon,
            variant="lda_robocasa",
        )
        logging.info("EXTRACT_SAVED %s: features %s, targets %s", out_path, feats.shape, targets.shape)

    return save


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    save_fn = add_feature_extraction(vla, args.episode_id, args.horizon, args.out)

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
        save_fn()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--episode_id", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--out", type=str, required=True)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    main(args)
