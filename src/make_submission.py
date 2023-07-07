import torch
import os
import subprocess
from torchvision import transforms
from Dataset.dataset import ETHDataset
from PIL import Image
from Model.model import ResidualAttentionUNet
from torchvision.utils import save_image

if __name__ == "__main__":
    base_path = "data/ethz-cil-road-segmentation-2023"
    image_path = os.path.join(base_path, "test/images")
    pad =  transforms.Compose([transforms.Pad(56, fill=(0)), transforms.ToTensor()])
    model = ResidualAttentionUNet(4,1)
    model.load_state_dict(torch.load("best_model_weights_new_model.pth"))
    model.eval()
    for file in os.listdir(image_path):
        print(file)
        image = Image.open(os.path.join(image_path, file))
        image = pad(image)
        mask = model(image.unsqueeze(0))
        mask = transforms.CenterCrop(400)(mask)
        save_image(mask.squeeze(0), f"submission/masks/{file}")


        