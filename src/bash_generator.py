import pandas as pd


def create_bash_script(model_name, encoder_name, encoder_weight):
    m_line = f"model_name={model_name}\n"
    en_line = f"encoder_name={encoder_name}\n"
    ew_line = f"encoder_weight={encoder_weight}\n"
    run_name_line = f"run_name=rs-$model_name-$encoder_name-$encoder_weight\n"

    with open("scripts/base_0.sh", "r") as f:
        base_0 = f.read()
    with open("scripts/base_1.sh", "r") as f:
        base_1 = f.read()

    with open(
        f"scripts/auto/{model_name}_{encoder_name}_{encoder_weight}.sh",
        "w",
    ) as f:
        f.write(base_0)
        f.write("\n")
        f.write(m_line)
        f.write(en_line)
        f.write(ew_line)
        f.write(run_name_line)
        print("\n")
        f.write(base_1)


model_names = [
    # "Unet",
    "UnetPlusPlus",
    # "MAnet",
    "Linknet",
    "FPN",
    # "PSPNet",
    # "PAN",
    # "DeepLabV3",
    "DeepLabV3Plus",
]
encoder_names = [
    "resnet34",
    "resnet50",
    "resnet101",
    # "resnet152",
    "efficientnet-b5",
    # "resnext50_32x4d",
    # "resnext101_32x8d",
    # "timm-resnest50d",
    # "timm-resnest101e",
]
encoder_weights = ["imagenet"]

for model_name in model_names:
    for encoder_name in encoder_names:
        for encoder_weight in encoder_weights:
            create_bash_script(model_name, encoder_name, encoder_weight)
