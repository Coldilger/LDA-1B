"""Experiment 3 (cost per decision) instrumentation for LDA-1B, RoboCasa harness.

Bridge's SimplerEnv wrapper (`lda_policy.py`) needs to gate timing on
`action_buffer is None` because the client-side loop calls `step()` on every
control tick and only some of those trigger a real replan. RoboCasa's client
(`robocasa-eval/launcher/run_client.py` -> the vendored
`examples.Robocasa_tabletop.eval_files.simulation_env`) already does its own
action-chunking client-side and only sends a websocket 'infer' request when
it actually needs a new chunk (`--args.n_action_steps 12`) -- so every
`predict_action` call the server receives already is one real per-decision
cost, not a cached-action pop. No gating needed here.

Wraps `vla.predict_action` with a wall-clock timer before handing it to
`WebsocketPolicyServer`, logging one `LATENCY_MS <value>` line per call to
this process's own stdout (captured in the job's `server-*.log`) -- does not
modify `server_policy.py`, `websocket_policy_server.py`, or the framework
itself. Mirrors F1-VLA/mimic-video's `timing_wrapper.py` pattern, adapted to
a class-decorator-on-an-instance since `predict_action` is called through a
long-lived server process rather than a fresh per-run script.
"""

import argparse
import logging
import socket
import time

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework


def add_timing(vla):
    original = vla.predict_action

    def timed(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logging.info("LATENCY_MS %.1f", dt_ms)
        return result

    vla.predict_action = timed
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
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
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
