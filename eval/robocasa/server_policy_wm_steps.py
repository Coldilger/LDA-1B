"""Experiment 9 (compute vs. performance, advisor's point 4) for LDA-1B:
the world-model-on policy of Experiment 1 (server_policy_worldmodel_on.py:
video_gen() imagines the next observation, then the inverse-dynamics task
predicts the action from it) with the number of denoising steps of the
forecast set by --vg_steps, and every decision timed (LATENCY_MS lines in
the server log, server_policy_timed.add_timing).

The checkpoint uses num_inference_timesteps = 4 for both the forecast and
the action; only the forecast's steps are changed here, the action is
always denoised with the checkpoint's own setting. --vg_steps 0 runs the
authors' default policy path (no forecast) through the same timer.

This varies the knob the LDA-1B pipeline exposes for its forecast, like the
denoising depth swept for mimic-video (Pai et al. 2025, Fig. 6). Note that
the world-model-on condition is our own composition of two trained tasks,
not an inference mode of the authors (thesis Section 3.3).
"""

import argparse
import logging
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework
from server_policy_timed import add_timing


def add_worldmodel_on_steps(vla, vg_steps: int):
    original_predict_action = vla.predict_action
    original_video_gen = vla.video_gen
    head = vla.action_model
    default_steps = head.num_inference_timesteps

    def worldmodel_predict_action(examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        head.num_inference_timesteps = vg_steps
        try:
            video_gen_out = original_video_gen(examples)
        finally:
            head.num_inference_timesteps = default_steps
        imagined_tokens = torch.from_numpy(video_gen_out["normalized_obs"])
        return original_predict_action(examples, inverse_dynamics_next_obs_tokens=imagined_tokens)

    vla.predict_action = worldmodel_predict_action
    logging.info("world model on: forecast steps %d, action steps %d", vg_steps, default_steps)
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    if args.vg_steps > 0:
        vla = add_worldmodel_on_steps(vla, args.vg_steps)
    vla = add_timing(vla)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, gpu: %s)", hostname, local_ip, torch.cuda.get_device_name(0))

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
    parser.add_argument("--vg_steps", type=int, required=True)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
