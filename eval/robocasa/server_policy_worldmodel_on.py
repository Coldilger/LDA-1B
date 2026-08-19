"""Experiment 1 for LDA-1B, real closed-loop: "turn the world model on where
it's normally off." Unlike server_policy_oracle_probe.py (a live L1 side
probe, useful as a fast first pass but not decisive -- offline/live L1 has
a memorization confound, see feedback_exp1_success_rate_metric), this makes
the world-model-on condition the actual policy driving the robot for full
episodes, so success rate -- the metric that actually counts for Experiment
1, matching F1-VLA/mimic-video's own "real closed-loop is the result that
matters" precedent -- can be measured the normal way.

Every real decision: call video_gen() to imagine the next observation, then
predict_action(..., inverse_dynamics_next_obs_tokens=<that>) to get the
action that's actually returned to the client and executed. No side
computation, no buffering across steps -- structurally the same as
server_policy.py, just with the extra video_gen()+inverse_dynamics hop
in between.
"""

import argparse
import logging
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework


def add_worldmodel_on(vla):
    original_predict_action = vla.predict_action
    original_video_gen = vla.video_gen

    def worldmodel_predict_action(examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        video_gen_out = original_video_gen(examples)
        imagined_tokens = torch.from_numpy(video_gen_out["normalized_obs"])
        return original_predict_action(
            examples, inverse_dynamics_next_obs_tokens=imagined_tokens
        )

    vla.predict_action = worldmodel_predict_action
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    vla = add_worldmodel_on(vla)

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
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
