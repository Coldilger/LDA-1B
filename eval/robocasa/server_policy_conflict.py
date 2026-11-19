"""Experiment 8 (instruction-foresight conflict, advisor's point 7) for
LDA-1B, in the world-model-on mode of Experiment 1 (video_gen() imagines the
next observation, the inverse-dynamics task predicts the action from it;
our composition of two trained tasks, not an inference mode of the
authors -- thesis Section 3.3).

Extends the cue-conflict design (Geirhos et al., 2019) and interchange
interventions (Geiger et al., 2021) as in F1-VLA's
eval/bridge/experiment8_conflict. The instruction reaches LDA's action head
by two routes: directly (the Qwen-VL features of image + instruction are the
MM-DiT's text tokens) and through the forecast (video_gen is conditioned on
the same features). The two are separable in code:
predict_action(examples, inverse_dynamics_next_obs_tokens=...) takes the
forecast as an argument.

Cells (A = the scene's instruction from the env, B = --alt text):
    AA   text A, forecast from A       (= the world-model-on policy)
    AA2  text A, forecast from A with another forecast seed (noise floor)
    BB   text B, forecast from B
    AB   text A, forecast from B
    BA   text B, forecast from A
Forecast seed and action-noise seed are fixed per decision and shared by
all cells, so differences come only from the text or the forecast.

--mode probe: the robot is driven by AA; every decision appends one JSON
    line with distances between the cells' action chunks to --records.
--mode cl: the robot is driven by the cell --text_src / --forecast_src.
"""

import argparse
import copy
import json
import logging
import socket

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lda.model.framework.base_framework import baseframework

def _prefix(text):
    # the env's instruction carries an embodiment prefix, e.g.
    # "unlocked_waist: pick up the cup, ..."; B keeps it so that A and B
    # differ only in the task words
    return text.split(": ", 1)[0] + ": " if ": " in text else ""


ALTS = {
    # same sentence, other object of the same task family (PnPCanToDrawerClose)
    "can": lambda a: a.replace("the cup", "the can") if "the cup" in a else _prefix(a) + "pick up the can, place it into the drawer and close the drawer",
    # a different goal in the same scene
    "drawer": lambda a: _prefix(a) + "close the drawer",
}


def _dist(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def add_conflict(vla, mode, text_src, forecast_src, alt_names, records):
    pa, vg = vla.predict_action, vla.video_gen
    state = {"call": 0, "logged": 0}

    def with_lang(examples, text):
        out = []
        for e in examples:
            e2 = copy.copy(e)
            e2["lang"] = text
            out.append(e2)
        return out

    def forecast(examples, seed):
        torch.manual_seed(seed)
        return torch.from_numpy(vg(examples)["normalized_obs"])

    def act(examples, fc, seed):
        torch.manual_seed(seed)
        return pa(examples, inverse_dynamics_next_obs_tokens=fc)

    def conflict_predict_action(examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        k = state["call"]
        state["call"] += 1
        f_seed, a_seed = 100_000 + 2 * k, 200_000 + k
        text_a = examples[0]["lang"]
        if mode == "cl":
            text_b = ALTS[alt_names[0]](text_a)
            ex = {"A": examples, "B": with_lang(examples, text_b)}
            fc = forecast(ex[forecast_src], f_seed)
            if state["logged"] < 2:
                logging.info("conflict-cl text=%s forecast=%s A=%r B=%r", text_src, forecast_src, text_a, text_b)
                state["logged"] += 1
            return act(ex[text_src], fc, a_seed)

        fA, fA2 = forecast(examples, f_seed), forecast(examples, f_seed + 1)
        out = act(examples, fA, a_seed)
        aa = out["normalized_actions"][0]
        aa2 = act(examples, fA2, a_seed)["normalized_actions"][0]
        rec = {"call": k, "lang": text_a, "d_AA2_AA": _dist(aa2, aa),
               "fc_seed_rel": float((fA2 - fA).norm() / (fA.norm() + 1e-8)), "alts": {}}
        for name in alt_names:
            text_b = ALTS[name](text_a)
            exB = with_lang(examples, text_b)
            fB = forecast(exB, f_seed)
            bb = act(exB, fB, a_seed)["normalized_actions"][0]
            ab = act(examples, fB, a_seed)["normalized_actions"][0]
            ba = act(exB, fA, a_seed)["normalized_actions"][0]
            d_ab_aa, d_ab_bb, d_ba_bb, d_ba_aa = _dist(ab, aa), _dist(ab, bb), _dist(ba, bb), _dist(ba, aa)
            rec["alts"][name] = {
                "instruction": text_b, "fc_rel_AB": float((fB - fA).norm() / (fA.norm() + 1e-8)),
                "d_BB_AA": _dist(bb, aa), "d_AB_AA": d_ab_aa, "d_BA_BB": d_ba_bb,
                # 0 = the action follows the text it reads, 1 = it follows the forecast
                "follow_AB": d_ab_aa / (d_ab_aa + d_ab_bb + 1e-12),
                "follow_BA": d_ba_bb / (d_ba_bb + d_ba_aa + 1e-12),
            }
        with open(records, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return out

    vla.predict_action = conflict_predict_action
    return vla


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()
    vla = add_conflict(vla, args.mode, args.text_src, args.forecast_src, args.alt.split(":"), args.records)
    logging.info("Creating server (host: %s)", socket.gethostname())
    server = WebsocketPolicyServer(policy=vla, host="0.0.0.0", port=args.port, idle_timeout=args.idle_timeout,
                                   metadata={"env": "simpler_env"})
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--idle_timeout", type=int, default=1800)
    p.add_argument("--mode", choices=["probe", "cl"], required=True)
    p.add_argument("--text_src", choices=["A", "B"], default="A")
    p.add_argument("--forecast_src", choices=["A", "B"], default="A")
    p.add_argument("--alt", type=str, default="can", help="probe: colon-separated ALTS keys; cl: one key")
    p.add_argument("--records", type=str, default=None)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
