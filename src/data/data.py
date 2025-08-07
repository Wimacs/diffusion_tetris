import os
from typing import List, Any, Tuple, Optional

import PIL.Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL

class SequencesDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        actions_path: str,
        seq_length: int,
        transform: Optional[Any] = None
    ) -> None:
        super().__init__()
        paths = sorted(
            [ item for item in os.listdir(images_dir) if item.endswith(".png")],
            key=lambda item: int(item.split(".")[0])
        )
        self.images_dir = images_dir
        with open(actions_path) as file:
            actions = [int(line) for line in file.readlines()]
        assert len(actions) == len(paths)
        self.sequences: List[Tuple[List[str], List[int]]] = []
        self.transform = transform
        for i in range(seq_length + 1, len(paths)):
            self.sequences.append((paths[max(i-seq_length - 1, 0) : i], actions[max(i - seq_length, 0) : i]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, actions = self.sequences[index]
        last_img = imgs[-1]
        def do_transform(img_path: str) -> Any:
            img_path = os.path.join(self.images_dir, img_path)
            image = PIL.Image.open(img_path).convert('RGB')
            if self.transform is not None:
                return self.transform(image)
            else:
                return image
        return (do_transform(last_img), torch.stack([do_transform(img) for img in imgs[:-1]]), torch.tensor(actions))
    
if __name__ == "__main__":
    import torchvision.transforms as transforms
    dataset = SequencesDataset(
        images_dir="training_data_directions/snapshots",
        actions_path="training_data_directions/actions",
        seq_length=3,
        transform=transforms.ToTensor()
    )
    dataset[0]