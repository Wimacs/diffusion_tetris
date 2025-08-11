import argparse
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import yaml

import pygame

from models.gen.blocks import UNet
from models.gen.edm import EDM
from data.data import SequencesDataset
import torchvision.transforms as transforms


def select_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    return device


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_unet_from_cfg(unet_cfg: dict, input_channels: int, actions_count: int, context_length: int) -> UNet:
    depths = unet_cfg.get("depths", unet_cfg.get("steps", [2, 2, 2, 2]))
    channels = unet_cfg["channels"]
    cond_channels = unet_cfg["cond_channels"]
    attn_depths = unet_cfg.get("attn_depths", unet_cfg.get("attn_step_indexes", [False] * len(channels)))
    return UNet(
        in_channels=input_channels * (context_length + 1),
        out_channels=input_channels,
        T=None,
        actions_count=actions_count,
        seq_length=context_length,
        depths=depths,
        channels=channels,
        cond_channels=cond_channels,
        attn_depths=attn_depths,
    )


def build_edm_from_cfg(model: UNet, edm_cfg: dict, context_length: int, device: str) -> EDM:
    s_tmax_cfg = edm_cfg.get("s_tmax", ".inf")
    s_tmax_val = float("inf") if str(s_tmax_cfg) in {".inf", "inf"} else float(s_tmax_cfg)
    return EDM(
        p_mean=edm_cfg["p_mean"],
        p_std=edm_cfg["p_std"],
        sigma_data=edm_cfg["sigma_data"],
        model=model,
        context_length=context_length,
        device=device,
        sigma_min=edm_cfg["sigma_min"],
        sigma_max=edm_cfg["sigma_max"],
        rho=edm_cfg["rho"],
        order=int(edm_cfg.get("order", 1)),
        s_churn=float(edm_cfg.get("s_churn", 0.0)),
        s_tmin=float(edm_cfg.get("s_tmin", 0.0)),
        s_tmax=s_tmax_val,
        s_noise=float(edm_cfg.get("s_noise", 1.0)),
        sigma_offset_noise=float(edm_cfg.get("sigma_offset_noise", 0.0)),
        quantize_output=bool(edm_cfg.get("quantize_output", True)),
    )


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_weights_into_unet(edm: EDM, ckpt_path: str, device: str) -> None:
    sd_all = torch.load(ckpt_path, map_location=device)
    sd = sd_all.get("model", sd_all)
    sd = strip_prefix(sd, "model.")
    sd = strip_prefix(sd, "module.")
    ret = edm.model.load_state_dict(sd, strict=False)
    print(ret)
    edm.model.eval()


def tensor_to_rgb(img: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] CHW tensor to uint8 HxWxC."""
    return (img.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def init_dataset(dataset_root: str, context_length: int):
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    dataset = SequencesDataset(
        images_dir=os.path.join(dataset_root, "snapshots"),
        actions_path=os.path.join(dataset_root, "actions"),
        seq_length=context_length,
        transform=transform_to_tensor
    )
    return dataset


ACTION_MAP_HELP = "0:No-op, 1:Up, 2:Down, 3:Left, 4:Right"


def get_current_action(pressed) -> int:
    """Return current action based on pressed keys. Priority: Up, Down, Left, Right, digits."""
    # Arrow keys
    if pressed[pygame.K_UP]:
        return 1
    if pressed[pygame.K_DOWN]:
        return 2
    if pressed[pygame.K_LEFT]:
        return 3
    if pressed[pygame.K_RIGHT]:
        return 4
    # Number keys
    if pressed[pygame.K_1] or pressed[pygame.K_KP1]:
        return 1
    if pressed[pygame.K_2] or pressed[pygame.K_KP2]:
        return 2
    if pressed[pygame.K_3] or pressed[pygame.K_KP3]:
        return 3
    if pressed[pygame.K_4] or pressed[pygame.K_KP4]:
        return 4
    # Explicit no-op
    if pressed[pygame.K_0] or pressed[pygame.K_KP0]:
        return 0
    return 0


def main():
    parser = argparse.ArgumentParser(description="EDM inference with pygame rendering")
    parser.add_argument("--config", type=str, default="config/Diffusion.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="tetris/data")
    parser.add_argument("--steps", type=int, default=30, help="denoising steps per frame")
    parser.add_argument("--scale", type=int, default=6, help="window scale factor for 64x64")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ui-fps", type=int, default=30, help="UI loop FPS")
    parser.add_argument("--autoplay-fps", type=int, default=5, help="autoplay sampling FPS")
    # Default autoplay enabled; user can disable with --no-autoplay
    parser.add_argument("--autoplay", dest="autoplay", action="store_true")
    parser.add_argument("--no-autoplay", dest="autoplay", action="store_false")
    parser.set_defaults(autoplay=True)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = select_device()
    cfg = load_config(args.config)
    gen_cfg = cfg["generation"]
    edm_cfg = cfg["edm"]

    input_channels = gen_cfg["input_channels"]
    context_length = gen_cfg["context_length"]
    actions_count = gen_cfg["actions_count"]

    unet = build_unet_from_cfg(edm_cfg["unet"], input_channels, actions_count, context_length)
    edm = build_edm_from_cfg(unet, edm_cfg, context_length, device)
    load_weights_into_unet(edm, args.ckpt, device)

    # Dataset and initial state
    dataset = init_dataset(args.dataset_root, context_length)
    assert len(dataset) > 0, "Empty dataset"

    # Pygame setup
    pygame.init()
    W, H = gen_cfg["image_size"], gen_cfg["image_size"]
    SCALE = args.scale
    screen = pygame.display.set_mode((W * SCALE, H * SCALE))
    pygame.display.set_caption(f"EDM Inference - {ACTION_MAP_HELP}")
    clock = pygame.time.Clock()

    running = True

    # Initialize sequence state
    def reset_sequence():
        idx = random.randint(0, len(dataset) - 1)
        img, last_imgs, actions = dataset[idx]
        return last_imgs.clone().to(device), actions.to(device)

    gen_imgs, actions_hist = reset_sequence()

    print("Controls:", ACTION_MAP_HELP, " | arrows/1-4 | a: toggle autoplay | s: single step | r: reset | esc/q: quit | default autoplay on")

    autoplay = args.autoplay
    last_sample_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    gen_imgs, actions_hist = reset_sequence()
                elif event.key == pygame.K_a:
                    autoplay = not autoplay
                elif event.key == pygame.K_s:
                    # single step using current pressed action
                    pressed = pygame.key.get_pressed()
                    act = get_current_action(pressed)
                    actions_hist = torch.concat((actions_hist, torch.tensor([act], device=device)))
                    with torch.no_grad():
                        next_img = edm.sample(
                            args.steps,
                            gen_imgs[0].shape,
                            gen_imgs[-context_length:].unsqueeze(0),
                            actions_hist[-context_length:].unsqueeze(0)
                        )[0]
                    gen_imgs = torch.concat([gen_imgs, next_img[None, :, :, :]], dim=0)

        # Autoplay stepping with current action each tick
        if autoplay:
            now = time.time()
            interval = 1.0 / max(1, args.autoplay_fps)
            if now - last_sample_time >= interval:
                pressed = pygame.key.get_pressed()
                act = get_current_action(pressed)
                actions_hist = torch.concat((actions_hist, torch.tensor([act], device=device)))
                with torch.no_grad():
                    next_img = edm.sample(
                        args.steps,
                        gen_imgs[0].shape,
                        gen_imgs[-context_length:].unsqueeze(0),
                        actions_hist[-context_length:].unsqueeze(0)
                    )[0]
                gen_imgs = torch.concat([gen_imgs, next_img[None, :, :, :]], dim=0)
                last_sample_time = now

        # Render last generated or context frame
        frame = gen_imgs[-1].detach().cpu()
        rgb = tensor_to_rgb(frame)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # pygame expects (W,H)
        surf = pygame.transform.scale(surf, (W * SCALE, H * SCALE))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(max(1, args.ui_fps))  # limit UI loop FPS

    pygame.quit()


if __name__ == "__main__":
    main() 
