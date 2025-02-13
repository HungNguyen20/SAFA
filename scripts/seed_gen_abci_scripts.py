import os
from pathlib import Path

dataset = "cifar10"
lr = 0.001
# noniid = "sparsity_0.85"

N = 200
K = 10

u = 0.1
alpha = 0.1
beta = N * 10

num_rounds = 10000
batch_size = 8

parameters = [(15, 0.1, 0.95)]
seeds = [1,2,3,4,5]

models = ["cnn"]
for model in models:
    algos = ["testv1", "feddc", "fedavg"]

    if not Path(f"./{dataset}/{model}/{N}_clients").exists():
        os.makedirs(f"./{dataset}/{model}/{N}_clients")


    header_text = "\
#!/bin/bash\n\
#$ -cwd\n\
#$ -l rt_G.small=1\n\
#$ -l h_rt=72:00:00\n\
#$ -o /home/aaa10078nj/Federated_Learning/folder-name/logs/cifar10/$JOB_NAME_$JOB_ID.log\n\
#$ -j y\n\n\
source /etc/profile.d/modules.sh\n\
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16\n\
source ~/venv/pytorch2023/bin/activate\n\
export PATH=$HOME/apps/openmpi/bin:$PATH\n\
python --version\n\
LOG_DIR=\"/home/aaa10078nj/Federated_Learning/folder-name/logs/cifar10/$JOB_NAME_$JOB_ID\"\n\
rm -r ${LOG_DIR}\n\
mkdir ${LOG_DIR}\n\
#Dataset\n\
DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
cp -r ../AAAI2023/easyFL/benchmark/cifar10/data ${DATA_DIR}\n\n\
"

    body_text = "\
python main.py \
--task ${TASK} \
--model ${MODEL} \
--algorithm ${ALG} \
--wandb ${WANDB} \
--data_folder ${DATA_DIR} \
--log_folder ${LOG_DIR} \
--dataidx_filename ${IDX_DIR} \
--num_rounds ${ROUND} \
--num_epochs ${EPOCH_PER_ROUND} \
--proportion ${PROPOTION} \
--batch_size ${BATCH} \
--learning_rate ${LR} \
--num_threads_per_gpu ${NUM_THRESH_PER_GPU} \
--num_gpus ${NUM_GPUS} \
--server_gpu_id ${SERVER_GPU_ID} \
--seed ${SEED}"

    formated_command = "\
GROUP=\"{}\"\n\
ALG=\"{}\"\n\
MODEL=\"{}\"\n\
WANDB=1\n\
ROUND={}\n\
LR={}\n\
EPOCH_PER_ROUND={}\n\
BATCH={}\n\
PROPOTION={:>.2f}\n\
NUM_THRESH_PER_GPU=1\n\
NUM_GPUS={}\n\
SERVER_GPU_ID=0\n\
TASK=\"{}\"\n\
IDX_DIR=\"{}/new/{}client/u{}_alpha{}_beta{}\"\n\
SEED={}\n\n\
cd MainFL\n\n\
"

    for E in [10]:
        folder_path = f"{dataset}/{model}/ablation_seeds/{N}_clients/"
        if not Path(folder_path).exists():
            os.makedirs(folder_path)
                    
        for algo in algos:
            
            for seed in seeds:
                task_name = f"{dataset}_{model}_seed{seed}_u{u}_alpha{alpha}_beta{beta}_N{N}_K{K}_E{E}_B{batch_size}"
                
                command = formated_command.format(
                    task_name, algo, model, num_rounds, lr, E, batch_size, K/N, 1, task_name, dataset, N, u, alpha, beta, seed
                )
                
                if algo == 'testv1':
                    for (offset, lambda_ft, mu) in parameters:
                        file = open(f"{folder_path}{task_name}_{algo}_offset{offset}_lambda{lambda_ft}_alpha{mu}.sh", "w")
                        new_command = f"OFFSET={offset}\nLAMBDA={lambda_ft}\nALPHA={mu}\n" + command
                        new_body_text = body_text + " --offset ${OFFSET} --lambda ${LAMBDA} --alpha ${ALPHA}"
                        file.write(header_text + new_command + new_body_text)
                        file.close()
                        
                elif algo == 'feddc':
                    file = open(f"{folder_path}{task_name}_{algo}_offset{offset}.sh", "w")
                    new_command = f"OFFSET={offset}\n" + command
                    new_body_text = body_text + " --offset ${OFFSET}"
                    file.write(header_text + new_command + new_body_text)
                    file.close()
                        
                else:
                    file = open(f"{folder_path}{task_name}_{algo}.sh", "w")
                    file.write(header_text + command + body_text)
                    file.close()

