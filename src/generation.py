from dataclasses import dataclass

@dataclass
class GenerationConfig:
    image_size: int
    input_channels: int
    output_channels: int
    context_length: int
    actions_count: int

    @property
    def unet_input_channels(self) -> int:
        return self.input_channels * (self.context_length + 1)