import torch
from PIL import Image
from torchvision.transforms import v2

REPO_DIR = "/home/juan/pyProjects/EUPE"
WEIGHTS = "/home/juan/pyProjects/EUPE/weights/EUPE-ViT-T.pt"


def get_img():
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image.show()
    return image


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


model = torch.hub.load(REPO_DIR, "eupe_vitt16", source="local", weights=WEIGHTS)

img_size = 256
img = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        outputs = model.forward_features(batch_img)
clstoken, patchtokens = outputs["x_norm_clstoken"], outputs["x_norm_patchtokens"]

print(clstoken.shape)
print(patchtokens.shape)
