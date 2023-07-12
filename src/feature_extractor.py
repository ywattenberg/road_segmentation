import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from trainer import Trainer
from Dataset.dataset import ETHDataset


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((400, 400)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225] )
])


if __name__ == "__main__":
    # dataset = torchvision.datasets.EuroSAT(root='data/', download=False, transform=transform)
    # print(dataset)
    # print(dataset.classes)

    base_path = "data/ethz-cil-road-segmentation-2023"
    image_path = os.path.join(base_path, "training/images")
    mask_path = os.path.join(base_path, "training/groundtruth")
    dataset = ETHDataset(image_path, mask_path, augment_images=False)

    # model = torchvision.models.mobilenet_v3_large()
    # print(model)
    #model.classifier[-1] = nn.Linear(1280, 10)

    model = torchvision.models.vgg13(pretrained=True)
    model.classifier = nn.Identity()
    model.avgpool = nn.Identity()
    model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    print(model)

    y = model.features[:9](dataset[0][0].unsqueeze(0))
    print(y.shape)

    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # trainer = Trainer(model, dataset, None, loss_fn, None, split_test=0.2, batch_size=16, epochs=20, test_metrics=[loss_fn], test_metric_names=["BinaryCrossEntropy"], epochs_between_safe=10, name="feature_extractor_transformer")
    # scores = trainer.train_test()
    # scores.to_csv("test_scores_feature_transformer.csv")
    
    
