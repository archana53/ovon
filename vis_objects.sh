#!/bin/sh -l
#SBATCH -p short
#SBATCH --gres=gpu:4
#SBATCH -J train_split_visualization
#SBATCH -o train_split_vis.log
#SBATCH -t 08:00:00
hostname
echo $CUDA_AVAILABLE_DEVICES
python ovon/dataset/visualise_objects.py -s "train" -f "/srv/cvmlp-lab/flash1/akutumbaka3/ovonproject/data/obj/filtered_raw_categories.json" 
