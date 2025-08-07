from typing import List
import os
import random

import click
import yaml
import pickle
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from training_utils.train_loop import train_loop, TrainingConfig
from generation import GenerationConfig
from models.gen.blocks import BaseDiffusionModel, UNet, UNetConfig
from models.gen.edm import EDM, EDMConfig
from models.gen.ddpm import DDPM, DDPMConfig
from utils.utils import EasyDict, instantiate_from_config
from data.data import SequencesDataset

def _save_sample_imgs(
    frames_real: torch.Tensor,
    frames_gen: List[torch.Tensor],
    path: str
):
    def get_np_img(tensor: torch.Tensor) -> np.ndarray:
        return (tensor * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    height_row = 5
    col_width = 5
    cols = len(frames_real)
    rows = 1 + len(frames_gen)
    fig, axes = plt.subplots(rows, cols, figsize=(col_width * cols, height_row * rows))
    for row in range(rows):
        frames = frames_real if row == 0 else frames_gen[row - 1]
        for i in range(len(frames_real)):
            axes[row, i].imshow(get_np_img(frames[i]))
            
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def _generate_and_save_sample_imgs(
    model: BaseDiffusionModel,
    dataset: SequencesDataset,
    epoch: int,
    device: str,
    context_length: int,
    length_session = 20
):
    if len(dataset) - 1 < length_session:
        length_session = len(dataset) - 1
    index = random.randint(0, len(dataset) - 1 - length_session)

    img, last_imgs, actions = dataset[index]

    img = img.to(device)
    last_imgs = last_imgs.to(device)
    actions = actions.to(device)

    real_imgs = last_imgs.clone()
    gen_2_imgs = last_imgs.clone()
    gen_10_imgs = last_imgs.clone()
    gen_5_imgs = last_imgs.clone()
    for j in range(1, length_session):
        img, last_imgs, actions = dataset[index + j]
        img = img.to(device)
        last_imgs = last_imgs.to(device)
        actions = actions.to(device)
        real_imgs = torch.concat([real_imgs, img[None, :, :, :]], dim=0)
        gen_img = model.sample(10, img.shape, gen_10_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
        gen_10_imgs = torch.concat([gen_10_imgs, gen_img[None, :, :, :]], dim=0)
        gen_img = model.sample(2, img.shape, gen_2_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
        gen_2_imgs = torch.concat([gen_2_imgs, gen_img[None, :, :, :]], dim=0)
        gen_img = model.sample(5, img.shape, gen_5_imgs[-context_length:].unsqueeze(0), actions.unsqueeze(0))[0]
        gen_5_imgs = torch.concat([gen_5_imgs, gen_img[None, :, :, :]], dim=0)

    _save_sample_imgs(real_imgs, [gen_10_imgs, gen_5_imgs, gen_2_imgs], f"val_images/{epoch}.png")

@click.command()
@click.option('--config', help='Config for training', metavar='YAML', type=str, required=True, default="config/Diffusion.yaml")
@click.option('--model-type', type=click.Choice(['ddpm', 'edm'], case_sensitive=False), default='edm')
@click.option('--output-prefix', help='Path to output, which used as prefix with suffix as epoch number', type=str, required=True)

@click.option('--dataset', help='Path to dataset', type=str, required=False)
@click.option('--output-loader', help='Path to save data loader', type=str, required=False)
@click.option('--loader', help='Path to dataloader', type=str, required=False)

@click.option('--gen-val-images', help='Generate validating images', is_flag=True, required=False, default=False)

@click.option('--last-checkpoint', help='Path of checkpoint to resume the training', type=str, required=False)
def main(**kwargs):
    options = EasyDict(kwargs)
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))
    
    training_config = TrainingConfig(**config.training)
    generation_config = GenerationConfig(**config.generation)

    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Mac OS
    if torch.backends.mps.is_available():
        device = "mps"

    model: BaseDiffusionModel
    if options.model_type == "edm":
        config = EDMConfig(**instantiate_from_config(config.edm))
        model = EDM.from_config(
            config=config,
            context_length=generation_config.context_length,
            device=device,
            model=UNet.from_config(
                config=config.unet,
                in_channels=generation_config.unet_input_channels,
                out_channels=generation_config.output_channels,
                actions_count=generation_config.actions_count,
                seq_length=generation_config.context_length
            )
        )
    elif options.model_type == "ddpm":
        config = DDPMConfig(**instantiate_from_config(config.ddpm))
        model = DDPM.from_config(
            config=config,
            context_length=generation_config.context_length,
            device=device,
            model=UNet.from_config(
                config=config.unet,
                in_channels=generation_config.unet_input_channels,
                out_channels=generation_config.output_channels,
                actions_count=generation_config.actions_count,
                seq_length=generation_config.context_length,
                T=config.T
            )
        )

    if options.dataset:
        dataset = SequencesDataset(
            images_dir=os.path.join(options.dataset, "snapshots"),
            actions_path=os.path.join(options.dataset, "actions"),
            seq_length=generation_config.context_length,
            transform=transform_to_tensor
        )

        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        valid_size = total_size - train_size

        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, valid_size]
        )

        if options.output_loader:
            with open(options.output_loader, 'wb') as f:
                pickle.dump({
                    "train": train_dataset,
                    "val": val_dataset,
                    "all": dataset
                }, f)
    elif options.loader:
        with open(options.loader, 'rb') as f:
            checkpoint = pickle.load(f)
            train_dataset = checkpoint['train']
            train_dataset.transform = transform_to_tensor
            val_dataset = checkpoint['val']
            val_dataset.transform = transform_to_tensor
            dataset = checkpoint['all']
            dataset.transform = transform_to_tensor

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers
    )

    def gen_val_images(epoch: int):
        _generate_and_save_sample_imgs(model, dataset, epoch, device, generation_config.context_length)

    print(f"Start training {options.model_type}")
    train_loop(
        model=model,
        device=device,
        config=training_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_path_prefix=options.output_prefix,
        existing_model_path=options["last_checkpoint"],
        gen_imgs=gen_val_images if options.gen_val_images else None
    )

if __name__ == "__main__":
    main()