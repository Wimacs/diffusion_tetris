import os
import yaml
import torch
from src.models.gen.blocks import UNet

def build_unet_from_cfg(unet_cfg, input_channels, actions_count, context_length):
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

def main():
    config_path = "config/Diffusion.yaml"
    ckpt_path = "models/model_9.pth"  # weight name
    onnx_path = "unet.onnx"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    gen_cfg = cfg["generation"]
    edm_cfg = cfg["edm"]
    input_channels = gen_cfg["input_channels"]
    context_length = gen_cfg["context_length"]
    actions_count = gen_cfg["actions_count"]
    img_size = gen_cfg["image_size"]

    # 构建并加载权重
    unet = build_unet_from_cfg(edm_cfg["unet"], input_channels, actions_count, context_length)
    sd_all = torch.load(ckpt_path, map_location="cpu")
    sd = sd_all.get("model", sd_all)
    for prefix in ("model.", "module."):
        if any(k.startswith(prefix) for k in sd.keys()):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    unet.load_state_dict(sd, strict=False)
    unet.eval()

    # 准备样例输入
    B = 1
    x = torch.randn(B, input_channels * (context_length + 1), img_size, img_size, dtype=torch.float32)
    t = torch.randn(B, 1, dtype=torch.float32)
    prev_actions = torch.zeros(B, context_length, dtype=torch.long)

    dynamic_axes = {
        "x": {0: "batch", 2: "height", 3: "width"},
        "t": {0: "batch"},
        "prev_actions": {0: "batch"},
        "y": {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        unet,
        (x, t, prev_actions),
        onnx_path,
        input_names=["x", "t", "prev_actions"],
        output_names=["y"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    main()