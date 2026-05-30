# Boosting Monocular Metric Depth Estimation via Bokeh Rendering

<p align="center">
  <b>ICML 2026</b>
</p>

<p align="center">
  <a href="https://fogradio.github.io/BokehDepth_Project/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.12425"><img src="https://img.shields.io/badge/Paper-ICML%202026-red" alt="Paper"></a>
  <a href="https://huggingface.co/fogradio/BokehDepth"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-BokehDepth-yellow" alt="Model"></a>
</p>

<p align="center">
  <a href="https://fogradio.github.io">Hangwei Zhang</a><sup>1,2</sup>&nbsp;&nbsp;
  <a href="https://atfortes.github.io/">Armando Fortes</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://wtybest.github.io/">Tianyi Wei</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://xingangpan.github.io">Xingang Pan</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>S-Lab, Nanyang Technological University&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Beihang University
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

> **BokehDepth** decouples bokeh synthesis from depth prediction and uses lens-aware defocus as a *supervision-free* geometric cue to improve the accuracy and physical consistency of monocular metric depth estimation. **Left:** conventional pipelines predict depth from a single sharp image and render bokeh from a noisy depth map. **Right:** our two-stage framework — Stage-1 generates a calibrated bokeh stack from a single image, and Stage-2 fuses the induced defocus cues to produce sharper, more reliable metric depth.

---

## Method

Bokeh and monocular depth are two sides of the same lens geometry, but existing methods use this link in only one direction. Conventional bokeh pipelines depend on a predicted depth map, so any depth error turns into a wrong blur radius or a broken occlusion edge. Monocular metric models, in turn, struggle on textureless and distant regions, which is exactly where defocus carries the strongest geometric signal. **BokehDepth** closes this loop. It treats synthetic defocus as a supervision-free geometric cue and feeds it back into the depth estimator through two stages.

<p align="center">
  <img src="assets/intro_wide.png" width="100%">
</p>

<p align="center">
  <em>(a) Standard monocular depth estimation maps a single RGB image to a depth map. (b) Classical bokeh rendering needs both the image and a depth map. (c) BokehDepth instead generates a calibrated bokeh stack from the image and uses the resulting defocus cues to sharpen depth estimation.</em>
</p>

<p align="center">
  <img src="assets/method.png" width="100%">
</p>

**Stage-1 — Physically Grounded Bokeh Generation.**
We build on **FLUX.1-Kontext**, a rectified-flow MMDiT backbone, and add a lightweight bokeh cross-attention adapter. Heterogeneous optical settings (focal length, aperture, focus distance) collapse into a single calibrated scalar `K` from the thin-lens circle-of-confusion model, which captures the near-linear relation between blur radius and disparity offset (`r ≈ K · Δdisp`). Conditioned on `K`, Stage-1 turns one sharp image into a multi-strength bokeh stack with no depth map at any point. A unified data pipeline aligns real defocused photos, synthetic renderings, and paired datasets onto this shared `K` axis.

**Stage-2 — Bokeh Stack Fusion for Depth.**
A **Divided Space Focus Attention (DSFA)** module is inserted into the ViT depth encoder. It first runs spatial attention inside each frame, conditioned on that frame's strength `K_f`, and then runs focus attention across frames at matching spatial locations, modulated by FiLM. Each location can therefore read how its blur grows with `K`, which is the physical depth-from-defocus cue. Only the reference-frame tokens are passed on, so the original DPT decoder and metric head stay untouched. DSFA is a plug-and-play addition that drops into strong depth foundations such as Depth Anything V2 and UniDepthV2.

> A *Depth-from-Bokeh Sweep* proposition shows this cue is principled: under calibrated control, regressing the bokeh radius on `K` across the stack gives an unbiased and consistent estimate of each pixel's inverse-depth offset, which recovers metric depth.


## Installation

We ship a one-shot installer that mirrors the `bokehdepth` conda environment we used during development. Everything (CUDA runtime included) is fetched from PyPI through the +cu128 wheels, so the host machine only needs a working NVIDIA driver and `conda`.

```bash
git clone https://github.com/fogradio/BokehDepth.git
cd BokehDepth
bash env/install.sh            # creates the `bokehdepth` env and wires LD_LIBRARY_PATH
conda activate bokehdepth       # ready to use
```

`env/install.sh` does three things:

1. Creates / refreshes the conda environment from `env/environment.yml` (Python 3.10, `ffmpeg`, and an optional GCC toolchain).
2. Installs every pip package listed in `env/requirements.txt`, which includes `torch==2.8.0+cu128`, `xformers`, `diffusers`, `transformers`, `accelerate`, and the rest of the BokehDepth stack.
3. Drops two scripts into `${CONDA_PREFIX}/etc/conda/{activate,deactivate}.d/` so that activating the env automatically prepends the right CUDA wheel directories (and the conda env's `lib/`) to `LD_LIBRARY_PATH`. This is what lets `run_inference.sh` stay clean.

> Override the env name with `CONDA_ENV=myenv bash env/install.sh`.
>
> If you also need the optional UniDepth CUDA extensions (only used for training losses, **not** for inference), build them after activation:
> ```bash
> cd UniDepth/unidepth/ops/knn            && python setup.py install && cd -
> cd UniDepth/unidepth/ops/extract_patches && python setup.py install && cd -
> ```

## Weights

Pretrained checkpoints live on the [`fogradio/BokehDepth`](https://huggingface.co/fogradio/BokehDepth) Hugging Face model card. Place them under `weights/` so that the inference script can pick them up with its defaults:

```bash
mkdir -p weights
huggingface-cli download fogradio/BokehDepth --local-dir weights/
```

The Stage-1 base model (`black-forest-labs/FLUX.1-Kontext-dev`) is downloaded on first use through `diffusers`; make sure your Hugging Face token has accepted the FLUX license.

### BokehMe weights (training only)

Stage-1 training relies on the BokehMe renderer to synthesise (image, target-bokeh) pairs on the fly. The training scripts expect the two BokehMe checkpoints at:

```
bokeh-generation/dataset/bokehme/checkpoints/arnet.pth
bokeh-generation/dataset/bokehme/checkpoints/iunet.pth
```

Download them from the official [BokehMe release](https://github.com/JuewenPeng/BokehMe) (the authors publish `arnet.pth` / `iunet.pth` alongside their inference code) and copy the two files to the paths above. Alternatively, point the trainer elsewhere with `--arnet_ckpt /path/to/arnet.pth --iunet_ckpt /path/to/iunet.pth`. Inference does **not** require these weights.

## Inference

With the env activated and the weights in place, run:

```bash
bash run_inference.sh
```

`run_inference.sh` reads its defaults from environment variables (override any of them inline). The interesting ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `REF_IMAGE` | `examples/ref.png` | Input RGB image |
| `K_VALUES` | `10.0 20.0 30.0` | Bokeh strengths used to build the defocus stack |
| `ADAPTER_CKPT` | `weights/bokeh_lora.bin` | Stage-1 LoRA adapter |
| `WEIGHTS_PATH` | `weights/UDv2_dsfa_release.pth` | Stage-2 UniDepthV2-DSFA checkpoint |
| `CONFIG_PATH` | `UniDepth/configs/config_v2_vitl14_DSFA_inference.json` | Stage-2 model config |
| `OUTPUT_ROOT` | `examples/` | Where per-run subdirectories are created |

Each run produces a timestamped directory under `OUTPUT_ROOT/` containing the Stage-1 defocus stack, the Stage-2 metric depth (`depth.npy` + `depth_color.png`), and a `pipeline_summary.json` recording every argument that was used.

## Training

We release the full training code for both stages and both Stage-2 DSFA variants. Stage-1 trains the bokeh-generation LoRA adapter on top of a frozen `FLUX.1-Kontext-dev`; Stage-2 trains DSFA depth fusion for UniDepthV2 and Depth Anything V2.

| Component | Training code | Launcher / config |
| --- | --- | --- |
| Stage-1 bokeh generation | `bokeh-generation/train_flux_I2I.py` | `bokeh-generation/train_flux_I2I.sh` |
| Stage-2 UniDepthV2-DSFA | `UniDepth/scripts/train_DSFA.py` | `UniDepth/scripts/dist_train_DSFA.sh`, `UniDepth/configs/config_v2_vitl14_DSFA_*.json` |
| Stage-2 Depth Anything V2-DSFA | `DepthAnythingV2/scripts/train_dsfa.py` | `DepthAnythingV2/scripts/dist_train_dsfa.sh`, `DepthAnythingV2/configs/dsfa_train.json` |

The Stage-1 script mixes T2I batches, with on-the-fly BokehMe synthesis, and I2I batches, with pre-rendered bokeh targets. The Stage-2 scripts consume calibrated defocus-stack metadata and train the DSFA fusion modules together with the depth backbone according to each variant's configuration.

### Stage-1 — Build bokeh-generation manifests

`train_flux_I2I.py` consumes one or more JSONL "manifests" listing the per-sample paths and the calibrated bokeh-strength conditioning value `dof_cond` (`K`). `bokeh-generation/build_manifest.py` turns a simple CSV table into such a manifest, without baking in any dataset-specific layout.

CSV templates are provided under `bokeh-generation/examples/`:

| File | Purpose |
| --- | --- |
| `examples/manifest_itw.csv` | In-the-wild Flickr-style samples used for **T2I + on-the-fly BokehMe synthesis**. Provide depth maps and foreground masks; bokeh targets are rendered at training time. |
| `examples/manifest_i2i.csv` | Paired (sharp, bokeh) captures used for **I2I**. Either supply EXIF fields (`N`, `fmm`, `f35mm`, `s1`, image dimensions) so the script computes `dof_cond`, or pass a pre-computed `dof_cond` directly. |

Fill in the CSVs with absolute paths, then convert them:

```bash
cd bokeh-generation
python build_manifest.py examples/manifest_itw.csv --output dataset/itw_dataset.jsonl
python build_manifest.py examples/manifest_i2i.csv --output dataset/i2i_dataset.jsonl
```

`build_manifest.py --help` documents every column. The most important ones:

- `input_image_path` (required) — absolute path to the source image.
- `target_image_path` — set this for I2I rows; leave blank for T2I rows.
- `dof_cond` *or* the EXIF tuple `(N, fmm, f35mm, s1)` — needed to compute the conditioning K value.
- `depth_map_path` / `fg_mask_path` — required for the T2I synthetic path; optional (and ignored) for I2I rows that already carry a target.
- `captions` — pipe-separated list of text prompts.

Multiple CSVs can be concatenated by listing them in order:

```bash
python build_manifest.py samples_a.csv samples_b.csv -o dataset/combined.jsonl
```

### Stage-1 — Launch bokeh-generation training

Pre-flight checks:

1. Ensure the BokehMe checkpoints described under [BokehMe weights (training only)](#bokehme-weights-training-only) are in place.
2. Edit `bokeh-generation/train_flux_I2I.sh` and point `ITW_JSONL` / `I2I_JSONLS` (and, optionally, `POST_BOKEME_JSONL`) at the manifests you generated above. Set `OUTPUT_DIR` to your desired checkpoint directory.

Then:

```bash
cd bokeh-generation
bash train_flux_I2I.sh
```

The shell script uses `accelerate launch` with `accelerate_config_4gpu.yaml` (4×GPU, bf16); adjust either the launcher arguments or the config file if your hardware differs. Each run writes a timestamped sub-directory under `OUTPUT_DIR/` containing accelerate checkpoints and wandb logs (set `--report_to none` to disable wandb).

The default configuration in `train_flux_I2I.sh` mirrors the Stage-1 LoRA release (`lora_rank=128`, `block_ids=0-56`, `--unfreeze_q`, `--unfreeze_k`, `prodigy` optimizer, 40 epochs). Inspect `train_flux_I2I.py --help` for the full list of flags, including variable-resolution training (`--variable_resolution`) and the BokehMe-failure online-synthesis mode (`--post_bokeme_syn`).

### Stage-2 — UniDepthV2-DSFA depth fusion

The Stage-2 training entrypoint is `UniDepth/scripts/train_DSFA.py`, launched via `UniDepth/scripts/dist_train_DSFA.sh`. Reference training configurations live in `UniDepth/configs/`; `config_v2_vitl14_DSFA_inference.json` is reserved for inference:

| Config | Train datasets | Notes |
| --- | --- | --- |
| `config_v2_vitl14_DSFA_nyuv2.json` | `NYUv2Depth` | Indoor setup; batch size 32; MSE loss enabled |
| `config_v2_vitl14_DSFA_hypersim.json` | `HyperSim` | Synthetic-supervision setup; batch size 32; MSE loss disabled |

Each manifest field should point to JSONL files produced by your dataset-preparation pipeline. `hypersim_manifest_paths` accepts a list of JSONL files and can also be overridden at launch with `HYPERSIM_MANIFEST_PATHS` or `HYPERSIM_MANIFEST_PATH`. See the in-code field documentation in `UniDepth/unidepth/datasets/{nyuv2,hypersim}.py`. The conditioning K value is read from the per-sample metadata, and the released configs select stack entries with `defocus_stack_indices: [0, 1, 2]`.

To run training on 4 GPUs:

```bash
cd UniDepth
bash scripts/dist_train_DSFA.sh configs/config_v2_vitl14_DSFA_nyuv2.json
```

Override the default 4-GPU launch via environment variables, e.g. `GPUS=2 SAVE_INTERVAL=500 bash scripts/dist_train_DSFA.sh ...`. Pre-trained backbone weights are loaded from `training.pretrained` and the pixel encoder's own `pretrained` key in the JSON — point both at your UniDepthV2 ViT-L/14 checkpoint before launching. Resuming a partial run only needs `RESUME_CKPT=/path/to/latest.pth bash scripts/dist_train_DSFA.sh ...`.

### Stage-2 — Depth Anything V2-DSFA depth fusion

The Depth Anything V2 DSFA release lives under `DepthAnythingV2/`. Spatial attention is applied within each stack frame, focus attention is applied across frames at each spatial location, and only the refined reference-frame tokens are decoded by the original DPT head.

The folder contains:

| Path | Purpose |
| --- | --- |
| `DepthAnythingV2/depth_anything_v2/dpt_dsfa.py` | Depth Anything V2 model with DSFA encoder fusion |
| `DepthAnythingV2/scripts/infer_dsfa.py` | Inference from a reference image plus a calibrated focus stack |
| `DepthAnythingV2/scripts/train_dsfa.py` | Training from JSON or JSONL manifests |
| `DepthAnythingV2/scripts/dist_train_dsfa.sh` | Accelerate launcher for multi-GPU training |
| `DepthAnythingV2/configs/` | Reference inference and training hyperparameters |

Depth Anything V2 DSFA accepts the same simple stack manifest shape used by BokehDepth Stage-1 outputs:

```json
{
  "ref": "/path/to/ref.png",
  "depth": "/path/to/depth.npy",
  "stack": ["/path/to/stack_0.png", "/path/to/stack_1.png"],
  "k": [10.0, 20.0]
}
```

Run inference from a manifest:

```bash
cd DepthAnythingV2
python scripts/infer_dsfa.py \
  --sample-path ../examples/run_xxx/manifest.jsonl \
  --checkpoint ../weights/DAV2_dsfa_release.pth \
  --outdir outputs/dav2_dsfa \
  --save-numpy
```

Or pass the image stack directly:

```bash
cd DepthAnythingV2
python scripts/infer_dsfa.py \
  --ref-image /path/to/ref.png \
  --stack-images /path/to/stack_0.png /path/to/stack_1.png \
  --k-values 10.0 20.0 \
  --checkpoint ../weights/DAV2_dsfa_release.pth
```

Train on one or more generated manifests:

```bash
cd DepthAnythingV2
python scripts/train_dsfa.py \
  --config configs/dsfa_train.json \
  --manifest-path /path/to/train_manifest.jsonl \
  --pretrained-from ../weights/depth_anything_v2_metric_vkitti_vitl.pth \
  --save-path outputs/train_dsfa
```

For distributed training:

```bash
cd DepthAnythingV2
MANIFEST_PATH=/path/to/train_manifest.jsonl \
PRETRAINED_FROM=../weights/depth_anything_v2_metric_vkitti_vitl.pth \
GPUS=4 \
bash scripts/dist_train_dsfa.sh
```

## TODO

- [x] Release model
- [x] Release inference code
- [x] Release Stage-1 (bokeh generation) training code
- [x] Release Stage-2 (depth fusion) training code


## Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhang2025bokehdepth,
  title={Boosting Monocular Metric Depth Estimation via Bokeh Rendering},
  author={Zhang, Hangwei and Fortes, Armando and Wei, Tianyi and Pan, Xingang},
  journal={arXiv preprint arXiv:2512.12425},
  year={2025}
}
```

## Acknowledgement

We build upon [BokehDiffusion](https://github.com/atfortes/BokehDiffusion), [FLUX.1-Kontext](https://github.com/black-forest-labs/flux), [BokehMe](https://github.com/JuewenPeng/BokehMe), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth). We would like to thank these projects that made this work possible.
