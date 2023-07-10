import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from trainer import Trainer


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((400, 400)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225] )
])


if __name__ == "__main__":
    # dataset = torchvision.datasets.EuroSAT(root='./data/EuroSAT_RGB', download=False, transform=transform)
    # print(dataset)
    # print(dataset.classes)

    model = torchvision.models.inception_v3(pretrained=True, progress=True)
    print(model)
    #model.classifier[-1] = nn.Linear(1280, 10)

    # model = torchvision.models.vit_b_16()
    # model.heads = nn.Linear(768, 10)

    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # trainer = Trainer(model, dataset, None, loss_fn, None, split_test=0.2, batch_size=16, epochs=20, test_metrics=[loss_fn], test_metric_names=["BinaryCrossEntropy"], epochs_between_safe=10, name="feature_extractor_transformer")
    # scores = trainer.train_test()
    # scores.to_csv("test_scores_feature_transformer.csv")
    
    