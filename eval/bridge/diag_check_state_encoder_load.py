#!/usr/bin/env python3
"""Verify the load_pretrained_backbones shape-mismatch-skip fix actually did
something, since smoke-v3-627158.out showed no "skipping N shape-mismatched
key(s)" warning at all -- either the fix silently worked with nothing to
report (fine) or state_encoder was never even compared (needs checking, not
assuming). Pure key/shape inspection, no model construction, no GPU needed.
"""

import torch

PRETRAIN = "/mnt/beegfsnew/scratch/3295540/hf_cache/hub/models--Wayer2--LDA-pretrain/blobs/607dc362b66a1e1083b64dc02a2e69a3d2d0307cf775e324fb1a377decdea3ce"
V3_SMOKE = "/mnt/beegfsnew/scratch/3295540/LDA-1B/outputs/lda_bridge_v3/bridge_finetune_v3/final_model/pytorch_model.pt"

print("Loading pretrain checkpoint (this is a 14G file, be patient)...")
pretrain_sd = torch.load(PRETRAIN, map_location="cpu")
print("Loading v3 smoke-test checkpoint...")
v3_sd = torch.load(V3_SMOKE, map_location="cpu")

pretrain_state_keys = {k: tuple(v.shape) for k, v in pretrain_sd.items() if "state_encoder" in k}
v3_state_keys = {k: tuple(v.shape) for k, v in v3_sd.items() if "state_encoder" in k}

print(f"\npretrain state_encoder keys: {len(pretrain_state_keys)}")
for k, s in sorted(pretrain_state_keys.items()):
    print(f"  {k}: {s}")

print(f"\nv3 smoke-test state_encoder keys: {len(v3_state_keys)}")
for k, s in sorted(v3_state_keys.items()):
    print(f"  {k}: {s}")

common = set(pretrain_state_keys) & set(v3_state_keys)
print(f"\ncommon keys: {len(common)}")
for k in sorted(common):
    match = "SAME SHAPE (unexpected -- would not need skipping)" if pretrain_state_keys[k] == v3_state_keys[k] else "DIFFERENT SHAPE (should have been skipped)"
    print(f"  {k}: pretrain={pretrain_state_keys[k]} vs v3={v3_state_keys[k]}  -- {match}")

# Also spot-check that a definitely-shared, non-state weight actually loaded
# from pretrain (not just randomly initialized) -- confirms the rest of the
# checkpoint really did transfer, not just "didn't crash".
spot_keys = [k for k in pretrain_sd if "vision_encoder" in k or "qwen" in k.lower()]
print(f"\nspot-checking a shared non-state weight transferred correctly...")
if spot_keys:
    k = spot_keys[0]
    if k in v3_sd:
        same = torch.equal(pretrain_sd[k].float(), v3_sd[k].float())
        print(f"  {k}: identical to pretrain? {same} (expect True -- this key should have loaded normally)")
    else:
        print(f"  {k}: not found in v3 checkpoint (unexpected)")
else:
    print("  no vision_encoder/qwen key found to spot-check")
