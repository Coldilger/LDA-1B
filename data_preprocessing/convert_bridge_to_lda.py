#!/usr/bin/env python
"""Convert bridge_orig_lerobot into the GR00T-LeRobot layout LDA's dataloader reads.

Source: IPEC-COMMUNITY/bridge_orig_lerobot (LeRobot v2.0, 53,192 episodes,
1,893,026 frames, 5 fps, WidowX, 256x256 videos).

The target layout is pinned by `OxeDataConfig` in
lda/dataloader/gr00t_lerobot/data_config.py, which is `class OxeDataConfig(BaseDataConfig): pass`
— i.e. exactly BaseDataConfig:

    state_keys  = left_eef_position(3) left_eef_rotation(3) left_gripper(1)
                  right_eef_position(3) right_eef_rotation(3) right_gripper(1)
    action_keys = same 14-dim bimanual layout
    video_keys  = ["video.top_head"]
    language    = ["annotation.language.action_text"]

Three things this conversion has to get right, all verified against the data
rather than assumed:

0. **`action.*_eef_position/rotation` must hold ABSOLUTE poses, not bridge's deltas.**
   The pretrain config sets `use_delta_action: true`, under which the loader calls
   `_get_delta_action_from_raw_data` -> `calculate_delta_eef` (utils/rotation_convert.py:67)
   on those columns. That function computes `dT_i = T_i^-1 @ T_{i+1}` — the relative
   transform expressed in the *gripper's own frame*. bridge_orig's `action` column is
   already a delta, but a **world-frame** one: it was verified here that
   `action_t[0:3] == state_{t+1}[0:3] - state_t[0:3]` to machine precision over 40
   episodes. Passing bridge's world-frame deltas through as if they were absolute
   poses would differentiate an already-differentiated signal *and* mix up two
   different frames — silent garbage, of exactly the kind that cost F1-VLA 30 points.
   So the action pose columns are filled with the absolute pose actually reached,
   `state_{t+1}[0:6]`, and LDA's own code derives the deltas in the convention its
   pretrained weights were trained on. The gripper is a command, not a pose, and is
   never differentiated — it is passed through from bridge's `action[6]`.

1. **The gripper is at index 7 of observation.state, not index 6.**
   bridge_orig's `observation.state` is 8-wide: xyz(3) + rpy(3) + pad(1) + gripper(1).
   Index 6 is a pad column that is identically 0.0 across the dataset (F1-VLA's
   stats dump shows std=0 for it). Slicing `state[:7]` — the obvious reading of
   "7-dof arm" — would feed the model a constant zero where the gripper belongs.
   `action` on the other hand IS 7-wide: dxyz(3) + drpy(3) + gripper(1), no pad.

2. **The right arm is zero-filled and that is safe.** WidowX is single-arm, so
   right_* columns are all zeros. StateActionTransform's q99 path explicitly
   masks the q01 == q99 case (transform/state_action.py:118-130) and passes such
   dims through unchanged, so zero-range columns neither divide by zero nor
   produce garbage. The unapply path multiplies by a zero range, mapping back to
   0 — also harmless.

Videos are not re-encoded or copied: `video.top_head` is symlinked to the source
`observation.images.image_0` directory, keeping the 21 GB of mp4s in one place.
image_0 is bridge's primary external camera and the view every episode has.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Widths of each compact-layout column, in the order BaseDataConfig lists them.
LIMB_KEYS = [
    ("left_eef_position", 3),
    ("left_eef_rotation", 3),
    ("left_gripper", 1),
    ("right_eef_position", 3),
    ("right_eef_rotation", 3),
    ("right_gripper", 1),
]

# Source slices. `None` means "zero-fill" (WidowX has no right arm).
# Note the state gripper slice: 7:8, skipping the pad column at index 6.
STATE_SRC = {
    "left_eef_position": slice(0, 3),
    "left_eef_rotation": slice(3, 6),
    "left_gripper": slice(7, 8),
    "right_eef_position": None,
    "right_eef_rotation": None,
    "right_gripper": None,
}
# Action pose columns are sourced from the *next* frame's absolute state (see
# point 0 in the module docstring); only the gripper comes from bridge's `action`.
ACTION_POSE_SRC = {
    "left_eef_position": slice(0, 3),
    "left_eef_rotation": slice(3, 6),
}
ACTION_GRIPPER_SRC = slice(6, 7)

PASSTHROUGH_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def _split(raw: np.ndarray, src: slice | None, width: int) -> np.ndarray:
    if src is None:
        return np.zeros((raw.shape[0], width), dtype=np.float32)
    return raw[:, src].astype(np.float32)


def convert_episode(src_parquet: Path, dst_parquet: Path) -> dict[str, np.ndarray]:
    """Rewrite one episode's parquet into split columns; return its values for stats."""
    df = pd.read_parquet(src_parquet)
    state = np.stack(df["observation.state"].values)
    action = np.stack(df["action"].values)

    # The absolute pose reached at t+1 is the action target for t. The final frame
    # has no successor, so it repeats the last observed pose (a zero-motion step,
    # which is what the episode's terminal action effectively is).
    next_state = np.concatenate([state[1:], state[-1:]], axis=0)

    out = {}
    collected = {}
    for name, width in LIMB_KEYS:
        s = _split(state, STATE_SRC[name], width)
        if name in ACTION_POSE_SRC:
            a = _split(next_state, ACTION_POSE_SRC[name], width)
        elif name == "left_gripper":
            a = _split(action, ACTION_GRIPPER_SRC, width)
        else:
            a = np.zeros((state.shape[0], width), dtype=np.float32)
        out[f"state.{name}"] = list(s)
        out[f"action.{name}"] = list(a)
        collected[f"state.{name}"] = s
        collected[f"action.{name}"] = a

    for col in PASSTHROUGH_COLUMNS:
        if col in df.columns:
            out[col] = df[col].values

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(dst_parquet, index=False)
    return collected


def convert_chunk(args) -> tuple[str, int, dict[str, np.ndarray]]:
    """Convert every episode parquet in one chunk directory."""
    src_root, dst_root, chunk_name = args
    src_root, dst_root = Path(src_root), Path(dst_root)
    src_chunk = src_root / "data" / chunk_name
    dst_chunk = dst_root / "data" / chunk_name

    per_key: dict[str, list[np.ndarray]] = {}
    n = 0
    for pq in sorted(src_chunk.glob("episode_*.parquet")):
        collected = convert_episode(pq, dst_chunk / pq.name)
        for k, v in collected.items():
            per_key.setdefault(k, []).append(v)
        n += 1

    merged = {k: np.concatenate(v, axis=0) for k, v in per_key.items()}
    return chunk_name, n, merged


def link_videos(src_root: Path, dst_root: Path, source_view: str) -> int:
    """Symlink video.top_head -> the source camera directory, per chunk."""
    linked = 0
    src_videos = src_root / "videos"
    for chunk_dir in sorted(src_videos.glob("chunk-*")):
        src_view = chunk_dir / source_view
        if not src_view.is_dir():
            continue
        dst_view = dst_root / "videos" / chunk_dir.name / "video.top_head"
        dst_view.parent.mkdir(parents=True, exist_ok=True)
        if dst_view.is_symlink() or dst_view.exists():
            dst_view.unlink()
        dst_view.symlink_to(src_view.resolve())
        linked += 1
    return linked


def compute_stats(values: dict[str, np.ndarray]) -> dict:
    """Aggregate q01/q99/mean/std/min/max per column — q99 is what the loader normalizes with."""
    stats = {}
    for key, arr in sorted(values.items()):
        stats[key] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }
    return stats


def write_modality_json(dst_root: Path) -> None:
    """Describe where each modality lives. start/end index *within* the named column."""
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for name, width in LIMB_KEYS:
        modality["state"][name] = {"start": 0, "end": width, "original_key": f"state.{name}"}
        modality["action"][name] = {"start": 0, "end": width, "original_key": f"action.{name}"}
    modality["video"]["top_head"] = {"original_key": "video.top_head"}
    # The loader resolves language through task_index -> tasks.jsonl, exactly as
    # the shipped demo_data does; no text is materialised into the parquet.
    modality["annotation"]["language.action_text"] = {"original_key": "task_index"}
    (dst_root / "meta" / "modality.json").write_text(json.dumps(modality, indent=4))


def write_info_json(src_root: Path, dst_root: Path, total_episodes: int, total_frames: int) -> None:
    info = json.loads((src_root / "meta" / "info.json").read_text())
    src_video_info = info["features"]["observation.images.image_0"]["info"]

    features = {}
    for name, width in LIMB_KEYS:
        for prefix in ("state", "action"):
            features[f"{prefix}.{name}"] = {"dtype": "float32", "shape": [width]}
    features["video.top_head"] = {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channels"],
        "info": src_video_info,
    }
    for col, dtype in [
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]:
        features[col] = {"dtype": dtype, "shape": [1], "names": None}

    info["features"] = features
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    (dst_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    ap.add_argument("--dst", default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--source-view", default="observation.images.image_0")
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Convert only the first N chunks — for a fast end-to-end dry run.",
    )
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)

    chunks = sorted(d.name for d in (src_root / "data").glob("chunk-*"))
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"chunks to convert: {len(chunks)}", flush=True)

    all_values: dict[str, list[np.ndarray]] = {}
    total_episodes = 0
    jobs = [(str(src_root), str(dst_root), c) for c in chunks]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_chunk, j): j[2] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            chunk_name, n, merged = fut.result()
            total_episodes += n
            for k, v in merged.items():
                all_values.setdefault(k, []).append(v)
            print(f"[{i}/{len(chunks)}] {chunk_name}: {n} episodes", flush=True)

    values = {k: np.concatenate(v, axis=0) for k, v in all_values.items()}
    total_frames = int(next(iter(values.values())).shape[0])
    print(f"converted {total_episodes} episodes / {total_frames} frames", flush=True)

    n_links = link_videos(src_root, dst_root, args.source_view)
    print(f"symlinked {n_links} chunk video dirs", flush=True)

    stats = compute_stats(values)
    # The loader reads stats_gr00t.json (datasets.py:66); stats.json is written
    # too so the directory stays readable by stock LeRobot tooling.
    (dst_root / "meta" / "stats_gr00t.json").write_text(json.dumps(stats, indent=2))
    (dst_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    shutil.copy(src_root / "meta" / "tasks.jsonl", dst_root / "meta" / "tasks.jsonl")

    # episodes.jsonl must list only what was actually converted, otherwise a
    # --max-chunks dry run advertises 53k episodes and the loader walks straight
    # into a missing parquet. On a full run this filter is a no-op.
    converted = {int(p.stem.split("_")[-1]) for p in (dst_root / "data").rglob("episode_*.parquet")}
    src_episodes = (src_root / "meta" / "episodes.jsonl").read_text().splitlines()
    kept = [ln for ln in src_episodes if ln.strip() and json.loads(ln)["episode_index"] in converted]
    (dst_root / "meta" / "episodes.jsonl").write_text("\n".join(kept) + "\n")
    print(f"episodes.jsonl: kept {len(kept)} of {len(src_episodes)}", flush=True)

    write_modality_json(dst_root)
    write_info_json(src_root, dst_root, total_episodes, total_frames)

    print("--- gripper sanity check (must NOT be constant zero) ---", flush=True)
    g = values["state.left_gripper"]
    print(f"state.left_gripper: min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
