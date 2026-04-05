from pathlib import Path

import requests
import torch
import torchvision.transforms.v2 as v2
from torch import nn
from tqdm import tqdm

REPO_DIR = "/home/juan/pyProjects/EUPE/"
BASE_URL = "https://huggingface.co/facebook/{model_name}/resolve/main/{filename}"

BASE_NAME = "EUPE-ViT-{name}"
size_to_name = {
    "tiny": "T",
    "small": "S",
    "base": "B",
}


def download_file(url: str, local_path: Path) -> Path:
    """Download a file from URL to local path if not cached."""
    if local_path.exists():
        print(f"Using cached file: {local_path}")
        return local_path

    print(f"Downloading {url}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            f.write(chunk)

    print(f"Downloaded to: {local_path}")
    return local_path


def load_model(size="tiny"):
    # Map size to model name suffix
    name_suffix = size_to_name.get(size, "T")
    model_name = BASE_NAME.format(name=name_suffix)

    local_dir = Path(f"./hf_models/{model_name}")
    weights_path = local_dir / f"{model_name}.pt"

    # Construct download URL
    url = BASE_URL.format(model_name=model_name, filename=f"{model_name}.pt")

    # Download with cache checking
    download_file(url, weights_path)
    print("Weights downloaded correctly!")

    model = torch.hub.load(REPO_DIR, "eupe_vitt16", source="local", weights=str(local_dir))
    # model = build_vit_model(size=size, weights_path=weights_path)
    print("Model loaded correctly!")

    return model


def get_img():
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


class EUPEViTEncoder(nn.Module):
    def __init__(
        self,
        size="tiny",
        input_size=256,
    ):
        super().__init__()
        self.model = load_model(size=size)

        self.input_size = input_size
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((input_size, input_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def forward(self, x):
        x = self.transforms(x)
        x = x if x.ndim > 3 else torch.unsqueeze(x, 0)
        assert x.ndim == 4, "Input shape must be [B, C, H, W]"

        outputs = self.model.forward_features(x)
        clstoken, patchtokens = outputs["x_norm_clstoken"], outputs["x_norm_patchtokens"]
        return clstoken, patchtokens


if __name__ == "__main__":
    model = EUPEViTEncoder()  #  load_model(size="tiny")

    img_size = 256
    img = get_img()
    # batch_img = transform(img)[None]

    outputs = model(img)

    print("Forward pass correctly!")
