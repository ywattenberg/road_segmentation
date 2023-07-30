#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=rs-finetune
#SBATCH --time=16:00:00
#SBATCH --output=/cluster/home/%u/road_segmentation/log/finetune-%j.out
#SBATCH --error=/cluster/home/%u/road_segmentation/log/finetune-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=8G
# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# rsync -ah --stats /cluster/scratch/$USER/wachterberg-street65.tar.gz $TMPDIR
# tar -xzf wachterberg-street65.tar.gz

# echo "Data copied at:     $(date)"

# Binary or script to execute
# load modules
module load gcc/8.2.0 python/3.11.2 cuda/11.8.0 cudnn/8.8.1.3 eth_proxy curl jq

echo "Dependencies installed"

cd $HOME/road_segmentation
model_name=UnetPlusPlus
encoder_name=efficientnet-b5
encoder_weight=imagenet
run_name="rs-UnetPlusPlus-efficientnet-b5-imagenet"
epochs=30
batch_size=32
learning_rate=0.0001
echo "Run name: ${run_name}"
echo "Model name: ${model_name}"
echo "Encoder name: ${encoder_name}"
echo "Encoder weight: ${encoder_weight}"
echo "Epochs: ${epochs}"
echo "Batch size: ${batch_size}"
echo "Learning rate: ${learning_rate}"

echo "Starting finetuning at:     $(date)"

bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Starting finetuning for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Model;${model_name};false" --field "Encoder;${encoder_name};false" --field "Encoder weight;${encoder_weight};false"

$HOME/road_segmentation/venv/bin/python3 $HOME/road_segmentation/src/finetune.py -m $model_name -en $encoder_name -ew $encoder_weight -e $epochs -bs $batch_size -lr $learning_rate

echo "Finished training at:     $(date)"

# discord notification on finish
bash $HOME/discord-webhook/discord.sh --webhook-url=https://discord.com/api/webhooks/1105789194959339611/-tDqh7eGfQJhaLoxjCsHbHrwTzhNEsR5SDxabXFiYdhg-KHwzN3kVwr87rxUggqWCQ0K --title "Finished finetuning for $USER" --color 3066993 --field "Date;$(date);false" --field "Jobid;${SLURM_JOB_ID};false" --field "Model;${model_name};false" --field "Encoder;${encoder_name};false" --field "Encoder weight;${encoder_weight};false"

# End the script with exit code 0
exit 0