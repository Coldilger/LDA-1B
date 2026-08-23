"""Experiments 1 and 2 (oracle injection / world-model-on) for LDA-1B, run as
a live side-probe during a normal RoboCasa closed-loop rollout.

No static RoboCasa training dataset is available on this cluster (the
checkpoint's own config points at `/data/RobotData/robocasa_1k_20hz`, which
doesn't exist here -- unlike Bridge, where `bridge_orig_lerobot` is present
locally and F1-VLA/mimic-video's own Exp2 offline probes read real logged
episodes from it). This substitutes live simulator rollouts for a static
dataset: at each real policy step, the *previous* step's (curr_img, the
policy's own action, which the client then executed) plus *this* step's real
image (the real "next" observation that resulted) form exactly the same kind
of (curr, next, real_action) triple F1/mimic's offline probes read from disk
-- generated online instead of read from disk, otherwise the same metric.

Two conditions probed per triple, mirroring MMDiT_ActionHeader.predict_action's
new oracle_future_imgs / inverse_dynamics_next_obs_tokens kwargs:
  - oracle: real next frame fed into the inverse_dynamics task (Experiment 2).
  - worldmodel: video_gen()'s own imagined next frame fed into the same task
    (Experiment 1 -- "turn the world model on where it's normally off").
Both compared by L1 against the real action the policy took (=~ground truth,
same role a logged dataset's `action` field plays for F1/mimic), alongside a
zero-action trivial baseline. Also compares oracle and world-model actions
directly against EACH OTHER (added 2026-08-22) -- oracle_l1/worldmodel_l1
are each a distance to the same third point (real_action), which does not
establish how close oracle and world-model are to one another; that's a
separate, previously-uncomputed number, needed to answer whether forecast
*correctness* matters here the way it does for F1-VLA's own copy of this
experiment (see ORACLE_EXPERIMENT.md's "Open: does correctness matter
here?" section for the full reasoning this closes).

Does not change what's actually served to the client: the probe is a pure
side computation using the model's own already-existing predict_action/
video_gen methods (this file adds no new model code -- see
lda/model/framework/QwenMMDiT.py and MMDiT_ActionHeader.py for the actual
oracle/video-gen hooks). A re-entrancy guard stops the side-calls (which
also go through predict_action) from re-triggering the probe recursively.
"""

import argparse
import logging
import socket
import statistics

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework


def add_oracle_probe(vla):
    original_predict_action = vla.predict_action
    original_video_gen = vla.video_gen

    state = {"prev_example": None, "prev_action": None, "in_probe": False}
    records = []  # each: dict(oracle_l1, worldmodel_l1, zero_l1, episode_idx)
    episode_idx = {"n": 0}
    episode_success = {}  # episode_idx -> bool, filled in by _probe_on_episode_end

    def l1(pred, real):
        return float(np.mean(np.abs(pred - real)))

    def probed_predict_action(examples, **kwargs):
        if state["in_probe"]:
            return original_predict_action(examples, **kwargs)
        if not isinstance(examples, list):
            examples = [examples]

        if state["prev_example"] is not None and state["prev_action"] is not None:
            state["in_probe"] = True
            try:
                prev_example = state["prev_example"]
                real_action = state["prev_action"]  # (T, action_dim), what was actually done

                # predict_action's oracle_future_imgs must be a single frame
                # (T=1) per view -- the multi-frame curr+history stack that
                # "image" normally carries would double-count into the
                # channel dim after MMDiT_ActionHeader's channel-folding
                # encode (see that file's own comment on this). Take the
                # most recent frame of the stack (assumed last -- deques
                # append newest at the end); an off-by-one on which index is
                # "most recent" would only bias the oracle's absolute
                # accuracy, not invalidate the ablated/shuffled-style
                # relative comparison this experiment is built on.
                this_frame = np.asarray(examples[0]["image"])[-1]
                oracle_future_imgs = np.array([[this_frame]])
                oracle_out = original_predict_action(
                    [prev_example], oracle_future_imgs=oracle_future_imgs
                )
                oracle_l1 = l1(oracle_out["normalized_actions"][0], real_action)

                video_gen_out = original_video_gen([prev_example])
                imagined_tokens = torch.from_numpy(video_gen_out["normalized_obs"])
                wm_out = original_predict_action(
                    [prev_example], inverse_dynamics_next_obs_tokens=imagined_tokens
                )
                wm_l1 = l1(wm_out["normalized_actions"][0], real_action)

                zero_l1 = l1(np.zeros_like(real_action), real_action)

                # The actually-decisive number for "does correctness matter
                # here": oracle_l1/wm_l1 above are each a distance to
                # real_action, not to each other -- both landing far from
                # real_action doesn't mean they're close to each other. Both
                # action vectors already exist above; just diff them
                # directly.
                oracle_vs_wm_l1 = l1(
                    oracle_out["normalized_actions"][0], wm_out["normalized_actions"][0]
                )

                records.append(dict(
                    oracle_l1=oracle_l1, worldmodel_l1=wm_l1, zero_l1=zero_l1,
                    oracle_vs_wm_l1=oracle_vs_wm_l1,
                    episode_idx=episode_idx["n"],
                ))
                logging.info(
                    "PROBE_SAMPLE n=%d ep=%d oracle_l1=%.5f worldmodel_l1=%.5f zero_l1=%.5f "
                    "oracle_vs_wm_l1=%.5f",
                    len(records), episode_idx["n"], oracle_l1, wm_l1, zero_l1, oracle_vs_wm_l1,
                )
            except Exception:
                logging.exception("Probe side-computation failed (real rollout unaffected)")
            finally:
                state["in_probe"] = False

        result = original_predict_action(examples, **kwargs)
        state["prev_example"] = examples[0]
        state["prev_action"] = result["normalized_actions"][0]
        return result

    def on_episode_end(success):
        # Reported by the RoboCasa client (simulation_env.py) right after an
        # episode's outcome is known -- see WebsocketClientPolicy's own
        # report_episode_end docstring. Records which episode just finished
        # succeeded, and clears the (prev_example, prev_action) pair so the
        # first predict_action call of the *next* episode is never compared
        # against an action taken at the tail of this one -- same guard
        # F1-VLA/mimic-video's own live-oracle probes apply on reset.
        episode_success[episode_idx["n"]] = bool(success)
        episode_idx["n"] += 1
        state["prev_example"] = None
        state["prev_action"] = None

    def summarize():
        for name in ("oracle", "worldmodel", "zero", "oracle_vs_wm"):
            key = f"{name}_l1"
            all_vals = [r[key] for r in records]
            succ_vals = [r[key] for r in records if episode_success.get(r["episode_idx"])]
            if not all_vals:
                continue
            logging.info(
                "PROBE_SUMMARY_ALL condition=%s n=%d mean=%.5f median=%.5f sd=%.5f",
                name, len(all_vals), statistics.mean(all_vals), statistics.median(all_vals),
                statistics.stdev(all_vals) if len(all_vals) > 1 else 0.0,
            )
            if succ_vals:
                logging.info(
                    "PROBE_SUMMARY_SUCCESSFUL_EPISODES_ONLY condition=%s n=%d mean=%.5f median=%.5f sd=%.5f",
                    name, len(succ_vals), statistics.mean(succ_vals), statistics.median(succ_vals),
                    statistics.stdev(succ_vals) if len(succ_vals) > 1 else 0.0,
                )
            else:
                logging.info(
                    "PROBE_SUMMARY_SUCCESSFUL_EPISODES_ONLY condition=%s: no successful-episode samples yet",
                    name,
                )

    original_probed = probed_predict_action

    def probed_predict_action_with_periodic_summary(examples, **kwargs):
        result = original_probed(examples, **kwargs)
        # Logged periodically, not just at shutdown -- the server process is
        # SIGTERM'd at the end of the eval slurm script, which does not run
        # Python `finally` blocks, so this is the reliable place to persist
        # partial results if the job is killed mid-episode.
        if len(records) and len(records) % 5 == 0:
            summarize()
        return result

    vla.predict_action = probed_predict_action_with_periodic_summary
    vla._probe_on_episode_end = on_episode_end
    vla._probe_summarize = summarize
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    vla = add_oracle_probe(vla)

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
        vla._probe_summarize()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
