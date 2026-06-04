# GLMap

This repository contains the core code for the paper *Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning*, which has been accepted as a Highlight at CVPR 2026. The official paper is available via *CVPR Open Access* at [this URL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Multi-Scale_Gaussian-Language_Map_for_Zero-shot_Embodied_Navigation_and_Reasoning_CVPR_2026_paper.html).

## Setup

### Conda Environment

```bash
conda env create -f environment.yml
conda activate glmap
```

### Gaussian Rasterization

```bash
git submodule update --init --recursive # To ensure the submodule (e.g. `glm`) is cloned
pip install third_party/diff-gaussian-rasterization
```

If rendering does not work in your environment, the demo script saves PLY files by default, which you can then open with online Gaussian splatting viewers such as <https://superspl.at/editor>.


### Habitat

#### Environment

Installing Habitat via Conda is also possible; here we choose a pip-based installation. After installation, you should see a `habitat_sim` package in your pip list.

```bash
cd third_party/habitat-lab
git checkout tags/v0.3.1 # To ensure compatibility with the demo
pip install -e habitat-lab
pip install -e habitat-baselines
```

#### Dataset

For dataset downloads, see <https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md>.

We recommend downloading the HM3D dataset from <https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d> and the MP3D dataset from <https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset>.

After the datasets are prepared, the data folder should look like the structure below. Minor differences in file structure are generally fine — for example, the symbolic links shown here are not required.

```bash
data
├── datasets
│   └── objectnav
├── scene_datasets
│   ├── hm3d -> ../versioned_data/hm3d-0.2/hm3d
│   ├── hm3d_v0.2 -> ./hm3d
│   └── mp3d
└── versioned_data
    └── hm3d-0.2
```

### MLLM

You can switch to a different MLLM by defining your own interface in `src/mllm`. By default, local models are called via Ollama. See <https://ollama.com/library/gemma3:27b> and <https://ollama.com/library/qwen3:8b> for setup and installation. Note that `gemma3:27b` requires at least 24GB VRAM (tested on RTX 3090).

### VLM

#### Install

See the [GroundingDINO Install Guide](https://github.com/IDEA-Research/GroundingDINO#hammer_and_wrench-install) for installation instructions.

```bash
pip install -e third_party/GroundingDINO
```

#### Model Weights Download

```bash
# mobile sam
wget -O data/mobile_sam.pt  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
# groundingdino
wget -O data/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# yolov7
wget -O data/yolov7-e6e.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```

#### Start VLM Services

We recommend starting GroundingDINO and other services as HTTP servers to speed up the launch process, avoiding the need to load models on every start.

```
python -m src.vlm.detector.grounding_dino --port 12181
python -m src.vlm.itm.blip2itm --port 12182
python -m src.vlm.segmentor.sam --port 12183
python -m src.vlm.detector.yolov7 --port 12184
```

By default, these environment variables are set to `true` (i.e., HTTP mode). You can switch to local (in-process) mode by setting them to `false`:

```
export VLM_GROUNDING_DINO_HTTP_MODE=false
export VLM_MOBILE_SAM_HTTP_MODE=false
export VLM_BLIP2ITM_HTTP_MODE=false
export VLM_YOLOV7_HTTP_MODE=false
```

## Manual Control Demo

### Launching the Demo

> Before launching, make sure that Ollama, VLM servers, and other services are already running.

We provide a manual control demo to showcase GLMap's incremental construction in Habitat, with the GLMap persisted to disk on exit.

Through this demo script, you can manually control the agent's movement using keyboard keys to perform incremental GLMap construction:

- `w` — Move forward
- `a` — Turn left
- `d` — Turn right
- `f` — Stop and finish

Note that the construction process can be slow due to MLLM inference calls.

By passing the `"--preset"` flag, the agent will follow a predefined action sequence (by default, spinning in place then stopping) to incrementally construct the GLMap without manual control. This is useful for quick debugging and verification.

```bash
python demo.py
```

You can switch datasets by specifying `config_name`, and set the target scene and episode by modifying the following code:

```python
target_scene, target_episode = "Dd4bFSTQ8gi", "0"
```

### Outputs

After the demo finishes, results are saved under `outputs/{scene}_{episode}/`:

- `glmap.pkl` — Serialized GLMap (instances, groups, obstacle map, agent path)
- `gaussians/` — Per-instance PLY files (`instance_{id}_{category}.ply`)
- `rendered/` — Rendered images (`instance_{id}_{category}.png`, `group_all.png`)

Intermediate results per step are saved under `outputs/{scene}_{episode}/{step}/`:

- `mllm_result.json` — Raw MLLM grounding output
- `segmented_img.png` — Visualization of detected objects

### Display Requirements

This script requires a display or X11 forwarding. Without one, you will see the following error:

```
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.
```

## Acknowledgements

In addition to the projects explicitly referenced in `third_party` (such as `diff-gaussian-rasterization`), we would also like to thank the following projects for their contributions:

- [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) for inspiring the manual control script.
- [VLFM](https://github.com/rai-opensource/vlfm) for VLM framework logic.

## Citation

If you find our work helpful for your research, please consider citing our paper:

```
@InProceedings{Zhang_2026_CVPR,
    author    = {Zhang, Sixian and Wang, Yiyao and Song, Xinhang and Zhang, Keming and Xu, Zijian and Jiang, Shuqiang},
    title     = {Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37086-37097}
}
```
