#!/bin/bash
# Multi-GPU launch script for train_flux_I2I.py.
#
# Edit the dataset paths and checkpoint locations near the bottom before use.
#
# Required dataset inputs (all CLI-configurable):
#   --itw_jsonl              Flickr-style in-the-wild JSONL (T2I + extra I2I source)
#   --i2i_jsonl              repeatable; pre-rendered I2I JSONLs (BLB / DPDD / Aperture / EBB / ...)
#   --post_bokeme_jsonl      (optional) JSONL of all-in-focus + depth pairs
#   --arnet_ckpt --iunet_ckpt
#                            BokehMe weights (default: dataset/bokehme/checkpoints/*.pth)

set -eo pipefail
conda activate bokehdepth
set -u


# -------  Edit these paths before launching -------
ITW_JSONL="/path/to/itw_dataset.jsonl"
I2I_JSONLS=(
  "/path/to/blb_i2i_dataset.jsonl"
  "/path/to/dpdd_i2i_dataset.jsonl"
  "/path/to/aperture_i2i_dataset.jsonl"
  "/path/to/ebb_i2i_dataset.jsonl"
)
POST_BOKEME_JSONL="/path/to/post_bokeme.jsonl"   # optional
OUTPUT_DIR="checkpoints"
ARNET_CKPT="dataset/bokehme/checkpoints/arnet.pth"
IUNET_CKPT="dataset/bokehme/checkpoints/iunet.pth"

# Convert the array into repeated --i2i_jsonl flags
I2I_ARGS=()
for p in "${I2I_JSONLS[@]}"; do
  I2I_ARGS+=(--i2i_jsonl "${p}")
done

# ------- Launch -------
accelerate launch \
  --config_file accelerate_config_4gpu.yaml \
  train_flux_I2I.py \
  --mixed_precision bf16 \
  --size 512 \
  --train_batch_size 4 \
  --lora_rank 128 \
  --lora_alpha 128 \
  --unfreeze_q \
  --unfreeze_k \
  --real_data_ratio 0.5 \
  --i2i_ratio 0.5 \
  --enable_grounded_attention \
  --itw_jsonl "${ITW_JSONL}" \
  "${I2I_ARGS[@]}" \
  --include_post_bokeme_syn \
  --post_bokeme_jsonl "${POST_BOKEME_JSONL}" \
  --arnet_ckpt "${ARNET_CKPT}" \
  --iunet_ckpt "${IUNET_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --block_ids="0-56" \
  --max_train_epochs=40 \
  --save_every_n_epochs=2 \
  --noise_offset=0.05 \
  --optimizer="prodigy" \
  --K_min=1.0 \
  --K_max=30.0
