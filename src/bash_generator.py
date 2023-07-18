import pandas as pd


def create_bash_script(model_name, encoder_name, encoder_weight):
    #!/bin/bash line
    bin_bash_line = "#!/bin/bash\n"
    # slurm settings
    mail_settings = "#SBATCH --mail-type=ALL\n"
    job_name = f"#SBATCH --job-name=rs-{model_name}\n"
    wall_time = f"#SBATCH --time=10:00:00\n"
    output = f"#SBATCH --output=/cluster/home/%u/road_segmentation/log/{model_name}-{encoder_name}-{encoder_weight}-%j.out\n"
    error = f"#SBATCH --error=/cluster/home/%u/road_segmentation/log/{model_name}-{encoder_name}-{encoder_weight}-%j.err\n"
    cpus = "#SBATCH --cpus-per-task=4\n"
    gpus = "#SBATCH --gpus=a100_80gb:1\n"
    mem = "#SBATCH --mem-per-cpu=8G\n"

    # model settings
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
        f.write(bin_bash_line)
        f.write(mail_settings)
        f.write(job_name)
        f.write(wall_time)
        f.write(output)
        f.write(error)
        f.write(cpus)
        f.write(gpus)
        f.write(mem)
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
