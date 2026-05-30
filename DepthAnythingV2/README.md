# Depth Anything V2 DSFA

This folder contains the Depth Anything V2 implementation of the DSFA depth fusion stage used by BokehDepth. It includes the model, a manifest-backed training entrypoint, and a single-image or manifest-backed inference entrypoint.

## Layout

| Path | Purpose |
| --- | --- |
| `depth_anything_v2/dpt_dsfa.py` | Depth Anything V2 with DSFA encoder fusion |
| `scripts/infer_dsfa.py` | Inference on a reference image plus a calibrated focus stack |
| `scripts/train_dsfa.py` | Training from JSON or JSONL manifests |
| `scripts/dist_train_dsfa.sh` | Accelerate launcher for distributed training |
| `datasets/defocus_stack.py` | Manifest dataset used by training |
| `configs/dsfa_inference.json` | Reference inference hyperparameters |
| `configs/dsfa_train.json` | Reference training hyperparameters |

## Manifest Format

Training and inference accept either JSON or JSONL. Each sample must provide a reference RGB image, a depth target for training, stack image paths, and the corresponding calibrated `K` values:

```json
{
  "ref": "/path/to/ref.png",
  "depth": "/path/to/depth.npy",
  "stack": ["/path/to/stack_0.png", "/path/to/stack_1.png"],
  "k": [10.0, 20.0]
}
```

The Stage-1 BokehDepth pipeline also writes a compatible `manifest.jsonl` for inference.

## Inference

```bash
cd BokehDepth/DepthAnythingV2
python scripts/infer_dsfa.py \
  --sample-path ../examples/run_xxx/manifest.jsonl \
  --checkpoint ../weights/DAV2_dsfa_release.pth \
  --outdir outputs/dav2_dsfa
```

You can also pass paths directly:

```bash
python scripts/infer_dsfa.py \
  --ref-image /path/to/ref.png \
  --stack-images /path/to/stack_0.png /path/to/stack_1.png \
  --k-values 10.0 20.0 \
  --checkpoint ../weights/DAV2_dsfa_release.pth \
  --save-numpy
```

## Training

Edit `configs/dsfa_train.json` or override its values on the command line:

```bash
cd BokehDepth/DepthAnythingV2
python scripts/train_dsfa.py \
  --config configs/dsfa_train.json \
  --manifest-path /path/to/train_manifest.jsonl \
  --pretrained-from ../weights/depth_anything_v2_metric_vkitti_vitl.pth \
  --save-path outputs/train_dsfa
```

For multi-GPU training:

```bash
cd BokehDepth/DepthAnythingV2
MANIFEST_PATH=/path/to/train_manifest.jsonl \
PRETRAINED_FROM=../weights/depth_anything_v2_metric_vkitti_vitl.pth \
GPUS=4 \
bash scripts/dist_train_dsfa.sh
```

The checkpoint is saved as `latest.pth` and `epoch_*.pth` under the selected save path.
