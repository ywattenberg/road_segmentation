#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job=rs-train
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/home/%u/road_segmentation/log/train-%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/cluster/home/%u/road_segmentation/log/train-%j.err  # where to store error messages
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=8G