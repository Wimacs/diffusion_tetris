from typing import Optional, Callable
import random
from dataclasses import dataclass
import os

import torch
import tqdm
from torch.utils.data.dataloader import DataLoader

from models.gen.blocks import BaseDiffusionModel

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    num_workers: int
    save_every_epoch: Optional[int] = None

def train_loop(
    model: BaseDiffusionModel,
    device: str,
    config: TrainingConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    output_path_prefix: str,
    existing_model_path: Optional[str] = None,
    gen_imgs: Callable[[int], None] = None
):
    if os.path.dirname(output_path_prefix) != "":
        os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    if gen_imgs:
        os.makedirs("val_images", exist_ok=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    epoch_range = range(1, config.epochs + 1)
    if existing_model_path is not None:
        parameters = torch.load(existing_model_path, map_location=device)
        model.load_state_dict(parameters["model"])
        optimizer.load_state_dict(parameters["optimizer"])
        epoch_range = range(parameters["epoch"] + 1, config.epochs + 1)

    training_losses = []
    val_losses = []
    for epoch in epoch_range:
        model.train(True)
        training_loss = 0
        val_loss = 0
        pbar = tqdm.tqdm(train_dataloader)
        for index, (imgs, previous_frames, previous_actions) in enumerate(pbar):
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            previous_frames = previous_frames.to(device)
            previous_actions = previous_actions.to(device)
    
            loss = model.forward(imgs, previous_frames, previous_actions)
    
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            pbar.set_description(f"loss for epoch {epoch}: {training_loss / (index + 1):.4f}")
        model.eval()
        with torch.no_grad():
            for index, (imgs, previous_frames, previous_actions) in enumerate(val_dataloader):
                imgs = imgs.to(device)
                previous_frames = previous_frames.to(device)
                previous_actions = previous_actions.to(device)
    
                loss = model(imgs, previous_frames, previous_actions)
        
                val_loss += loss.item()
                pbar.set_description(f"val loss for epoch {epoch}: {val_loss / (index + 1):.4f}")
            if gen_imgs:
                gen_imgs(epoch)
        training_losses.append(training_loss / len(val_dataloader))
        val_losses.append(val_loss / len(val_dataloader))
        if config.save_every_epoch is not None and epoch > 0 and epoch % config.save_every_epoch == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, f"{output_path_prefix}_{epoch}.pth")
    return training_losses, val_losses