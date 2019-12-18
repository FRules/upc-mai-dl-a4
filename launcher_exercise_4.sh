#!/bin/bash

#SBATCH --job-name="mnist_multi_gpu"
#SBATCH --qos=debug
#SBATCH --workdir=.
#SBATCH --output=mnist_multi_gpu_%j.out
#SBATCH --error=mnist_multi_gpu_%j.err
#SBATCH --ntasks=4
#SBATCH --gres gpu:4
#SBATCH --time=00:10:00

modulegpurge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python train.py
ad merovingian
module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
python exercise_4.py