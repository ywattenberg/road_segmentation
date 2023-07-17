import sys
import os
import click


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
    default="resnet34",
    type=str,
)
@click.option(
    "--encoder-weights",
    "-ew",
    help="Which encoder weights to use",
    default="imagenet",
    type=str,
)
def main(model_name, encoder_name, encoder_weights):
    # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    base_path = "/cluster/scratch/siwachte/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    skeleton_path = os.path.join(base_path, "skel")
    dataset = GMapsDataset(
        image_path, mask_path, skel_path=skeleton_path, augment_images=True
    )
    model_func = getattr(smp, model_name)

    model = model_func(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        activation="sigmoid",
        in_channels=4,
        classes=1,
    )
    loss_fn = SoftDiceClDice(0.5)
    optimizer = Lion(model.parameters(), lr=1e-3, weight_decay=1e-3)
    trainer = Trainer(
        model,
        dataset,
        None,
        loss_fn,
        None,
        split_test=0.2,
        batch_size=1,
        epochs=10,
        test_metrics=[JaccardLoss(mode="binary"), DiceLoss(mode="binary"), loss_fn],
        test_metric_names=["JaccardLoss", "DiceLoss", "clDice"],
        epochs_between_safe=1,
        name=f"{model_name}-{encoder_name}-{encoder_weights}-clDice",
    )

    scores = trainer.train_test()
    scores.to_csv(f"smp/{model_name}-{encoder_name}-{encoder_weights}-clDice.csv")


if __name__ == "__main__":
    main()
