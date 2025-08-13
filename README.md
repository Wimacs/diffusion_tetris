<p align="center">
  <video src="media/EDM%20Inference%20-%200_No-op,%201_Up,%202_Down,%203_Left,%204_Right%202025-08-13%2014-37-22.mp4" controls muted playsinline width="640"></video>
  <br/>
  <a href="media/EDM%20Inference%20-%200_No-op,%201_Up,%202_Down,%203_Left,%204_Right%202025-08-13%2014-37-22.mp4">Download/view sample video</a>
  
</p>

## Diffusion Tetris

A conditional diffusion model that predicts the next 64x64 RGB frame of a Tetris-like game given the previous context frames and a sequence of player actions. The repository includes:

- Training pipeline (EDM or DDPM) with PyTorch
- Interactive inference viewer with pygame
- ONNX export and a browser demo powered by onnxruntime-web

### Highlights
- **Conditioned generation**: next frame conditioned on last N frames and discrete actions in {0: noop, 1: up/rotate, 2: down, 3: left, 4: right}
- **Two samplers**: EDM (Karras sigmas) and DDPM
- **Web demo**: run sampling fully in the browser with an exported `unet.onnx`

## Repository Structure
- `src/`
  - `train.py`: CLI for training (EDM/DDPM)
  - `infer_pygame.py`: interactive inference viewer (pygame)
  - `generation.py`: dataclass for generation config
  - `models/`: UNet, EDM, DDPM implementations
  - `training_utils/`: training loop utilities
  - `data/`: dataset loader (`SequencesDataset`)
  - `utils/`: small helpers (`EasyDict`, config instantiation)
- `config/Diffusion.yaml`: training and generation configuration
- `models/`: checkpoints (e.g., `model_9.pth`)
- `docs/`: static web demo (ONNX + JS + HTML)
- `export_onnx.py`: export trained UNet to ONNX for web inference
- `tetris/`: a simple pygame Tetris implementation and example recorded data

## Requirements
Install Python dependencies (PyTorch, torchvision, and common utilities). PyTorch install varies by CUDA; visit the official site if needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA/CPU
pip install click pyyaml pillow numpy matplotlib tensorboard onnx onnxruntime
# Optional (for pygame-based viewers/tools):
pip install -r requirements.txt  # contains pygame
```

## Dataset Format
The training loader expects a directory with PNG frames and a matching actions file:

```
DATASET_ROOT/
  snapshots/
    0.png
    1.png
    ...
  actions
    # plain text; one integer per line (0..4), same length as snapshots
```

- Images should be 64x64 RGB normalized during training as `(x-0.5)/0.5`.
- Actions must align 1:1 with frames. The loader constructs sliding windows of length `context_length+1`.

An example dataset is provided at `tetris/data` (large files may be zipped as `tetris/data.zip`). If you record your own data, ensure you downscale frames to 64x64 and produce the `actions` file as described.

## Training
Edit `config/Diffusion.yaml` to adjust training, model, and sampler parameters. Then run:

```bash
python -m src.train \
  --config config/Diffusion.yaml \
  --model-type edm \
  --dataset tetris/data \
  --output-prefix models/model \
  --gen-val-images \
  --logdir runs/edm
```

Notes:
- Use `--model-type ddpm` to train a DDPM variant.
- Resume from a checkpoint with `--last-checkpoint models/model_9.pth`.
- `val_images/` will be populated when `--gen-val-images` is set.

## Interactive Inference (pygame)
Run a local interactive viewer that samples the next frame and renders it:

```bash
python -m src.infer_pygame \
  --config config/Diffusion.yaml \
  --ckpt models/model_9.pth \
  --dataset-root tetris/data \
  --steps 30 \
  --scale 6
```

Controls during the viewer:
- Arrows or digits 1..4: choose action (Up/Down/Left/Right)
- 0: no-op
- A: toggle autoplay; S: single sampling step; R: reset; Esc/Q: quit

## Export to ONNX
Export the trained UNet to ONNX for browser inference:

```bash
python export_onnx.py
```



Usage:
- Drop or auto-load `unet.onnx` (place it in `docs/` or update `docs/config.json` → `model_url`)
- Optionally load a start image; otherwise the sequence starts from black frames
- Use the same keyboard controls as the pygame viewer (arrows 1..4, 0, A/S/R)

`docs/config.json` mirrors key parts of `config/Diffusion.yaml` (generation size, EDM schedule) and points to the ONNX file. The browser sampler matches the PyTorch EDM path (order=1 by default).

## Configuration Reference
`config/Diffusion.yaml` contains:
- `training`: epochs, batch size, workers, checkpoint interval
- `generation`: `image_size`, `input_channels`, `output_channels`, `context_length`, `actions_count`
- `edm`: `p_mean`, `p_std`, Karras schedule (`sigma_min`, `sigma_max`, `rho`), integration `order`, `s_*` params, `sigma_offset_noise`, `quantize_output`, and the UNet sub-config
- `ddpm`: `T` and a UNet sub-config


