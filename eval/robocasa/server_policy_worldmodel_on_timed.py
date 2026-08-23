"""Experiment 3 (cost per decision) instrumentation for LDA-1B's world-model-ON
condition -- the number that never existed. `experiment3_cost/README.md`'s
headline 254.7ms is the DEFAULT path's cost (world-model dormant, per that
doc's own text: "the `policy` inference path... never runs `video_gen`/
`inverse_dynamics` at all, so there's no extra world-model compute to
measure"). `server_policy_worldmodel_on.py` (Experiment 1's real
closed-loop world-model-on driver, 44% vs 48% baseline) has zero timing
instrumentation -- built only to measure success rate, not latency.

Purely compositional, touches neither existing script: reuses
`add_worldmodel_on` (from server_policy_worldmodel_on.py, unmodified --
redefines predict_action to do video_gen() -> inverse_dynamics) and
`add_timing` (from server_policy_timed.py, unmodified -- wall-clock wrapper
+ LATENCY_MS logging), composed in that order so the timer measures the
*entire* world-model-on call, video_gen included.
"""

import argparse
import logging
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework
from server_policy_timed import add_timing
from server_policy_worldmodel_on import add_worldmodel_on


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    vla = add_worldmodel_on(vla)
    vla = add_timing(vla)

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
