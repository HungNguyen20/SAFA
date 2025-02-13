#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Hung_FCL/logs/cifar100/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16
source ~/venv/pytorch2023/bin/activate
export PATH=$HOME/apps/openmpi/bin:$PATH
python --version
LOG_DIR="/home/aaa10078nj/Federated_Learning/Hung_FCL/logs/cifar100/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}
#Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ../AAAI2023/easyFL/benchmark/cifar100/data ${DATA_DIR}

GROUP="cifar100_resnet9_u0.3_alpha1.5_beta5000_N500_K50_E10_B8"
ALG="fedfv"
MODEL="resnet9"
WANDB=1
ROUND=4000
LR=0.1
EPOCH_PER_ROUND=10
BATCH=8
PROPOTION=0.10
NUM_THRESH_PER_GPU=1
NUM_GPUS=1
SERVER_GPU_ID=0
TASK="cifar100_resnet9_u0.3_alpha1.5_beta5000_N500_K50_E10_B8"
IDX_DIR="cifar100/new/500client/u0.3_alpha1.5_beta5000"

cd MainFL

python main.py --task ${TASK} --model ${MODEL} --algorithm ${ALG} --wandb ${WANDB} --data_folder ${DATA_DIR} --log_folder ${LOG_DIR} --dataidx_filename ${IDX_DIR} --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --learning_rate ${LR} --num_threads_per_gpu ${NUM_THRESH_PER_GPU} --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID}