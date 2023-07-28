import sys
import os
import click
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from lion_pytorch import Lion

from Dataset.dataset import ETHDataset, MassachusettsDataset, GMapsDataset
from Loss.cldice import SoftDiceClDice
from trainer import Trainer


@click.command()
@click.option(
    "--model-name",
    "-m",
    help="Which model to use",
    default="UnetPlusPlus",
    type=str,
)
@click.option(
    "--encoder-name",
    "-en",
    help="Which encoder to use",
    default="efficientnet-b5",
    type=str,
)
@click.option(
    "--encoder-weights",
    "-ew",
    help="Which encoder weights to use",
    default="imagenet",
    type=str,
)
@click.option(
    "--epochs",
    "-e",
    help="Number of epochs to train for",
    default=5,
    type=int,
)
@click.option(
    "--batch-size",
    "-bs",
    help="Batch size",
    default=16,
    type=int,
)
@click.option(
    "--learning-rate",
    "-lr",
    help="Learning rate",
    default=1e-4,
    type=float,
)
@click.option(
    "--base-path",
    "-bp",
    help="Base path for data",
    default="data/ethz-cil-road-segmentation-2023/training",
    type=str,
)
@click.option(
    "--device",
    "-d",
    help="Device to use",
    default="cuda",
    type=str,
)
def main(
    model_name,
    encoder_name,
    encoder_weights,
    epochs,
    batch_size,
    learning_rate,
    base_path,
    device,
):
    # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "groundtruth")
    skel_path = os.path.join(base_path, "skel")
    dataset = ETHDataset(image_path, mask_path, skel_path, augment_images=True)
    model_func = getattr(smp, model_name)

    model = model_func(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        # activation="sigmoid",
        in_channels=3,
        classes=1,
    )
    print(
        f"Loading model: best_model_weights_{model_name}-{encoder_name}-{encoder_weights}-clDice.pth"
    )
    model.load_state_dict(
        torch.load(
            f"models/best_model_weights_{model_name}-{encoder_name}-{encoder_weights}-clDice.pth",
            map_location=torch.device(device),
        )
    )

    loss_fn = SoftDiceClDice(0.5)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    trainer = Trainer(
        model,
        dataset,
        None,
        loss_fn,
        optimizer,
        split_test=0.2,
        batch_size=batch_size,
        epochs=epochs,
        test_metrics=[JaccardLoss(mode="binary"), DiceLoss(mode="binary"), loss_fn],
        test_metric_names=["JaccardLoss", "DiceLoss", "clDice"],
        epochs_between_safe=1,
        name=f"{model_name}-{encoder_name}-{encoder_weights}-finetuned",
        device=device,
    )

    # for i in range(len(dataset)):
    #     dataset[i]
    scores = trainer.train_test()
    scores.to_csv(
        f"finetune/{model_name}-{encoder_name}-{encoder_weights}-finetuned.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
