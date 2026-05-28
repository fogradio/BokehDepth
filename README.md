# Boosting Monocular Metric Depth Estimation via Bokeh Rendering

<p align="center">
  <b>ICML 2026</b>
</p>

<p align="center">
  <a href="https://fogradio.github.io/BokehDepth_Project/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.12425"><img src="https://img.shields.io/badge/Paper-ICML%202026-red" alt="Paper"></a>
  <a href="https://huggingface.co/fogradio/BokehDepth"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Coming%20Soon-yellow" alt="Model"></a>
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


## TODO

- [ ] Release model
- [ ] Release inference code
- [ ] Release training code


## Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhang2025bokehdepth,
  title={Boosting Monocular Metric Depth Estimation via Bokeh Rendering},
  author={Zhang, Hangwei and Fortes, Armando Teles and Wei, Tianyi and Pan, Xingang},
  journal={arXiv preprint arXiv:2512.12425},
  year={2025}
}
```

## Acknowledgement

We build upon [BokehDiffusion](https://github.com/atfortes/BokehDiffusion), [FLUX.1-Kontext](https://github.com/black-forest-labs/flux), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth). We would like to thank these projects that made this work possible.


