"""One-off diagnostic: which index of examples[0]["image"]'s multi-frame
stack is the newest frame? Logs each frame's mean pixel value per real
step, tagged with a step counter, so consecutive calls can be compared by
hand: if index -1 is newest, this_step[i] should closely match
prev_step[i+1] for i in range(T-1) (the window slides forward by one), and
vice versa if index 0 is newest. Does not change what's served -- pure
logging wrapper around the unmodified policy path.
"""

import argparse
import logging
import socket

import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework


def add_frame_diag(vla):
    original_predict_action = vla.predict_action
    step = {"n": 0}

    def diag_predict_action(examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        img = np.asarray(examples[0]["image"])  # (T, H, W, C)
        means = [float(img[t].mean()) for t in range(img.shape[0])]
        step["n"] += 1
        logging.info("FRAME_DIAG step=%d T=%d frame_means=%s", step["n"], img.shape[0], means)
        return original_predict_action(examples, **kwargs)

    vla.predict_action = diag_predict_action
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    vla = add_frame_diag(vla)

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
