import os

visible_cudas = [0, 1]
cudas = ",".join([str(i) for i in visible_cudas])
task_file = "main.py"

dataset = "mnist"
noniid = "pareto"
N = 100
K = 10
total_epochs = 5000
batch_size = 16

model = "cnn"
algos = ["scaffold", "mp_proposal", "mp_fedavg"]
data_folder = f"./benchmark/{dataset}/data"
log_folder = f"motiv/{dataset}"

for E in [1, 5, 10, 20, 25, 40, 50]:
    task_name = f"{dataset}_{noniid}_N{N}_K{K}"

    for algo in algos:
        formated_command = f"CUDA_VISIBLE_DEVICES={cudas}\
            python main.py \
            --task {task_name} \
            --model {model} \
            --algorithm {algo} \
            --wandb 1 \
            --data_folder \"{data_folder}\" \
            --log_folder \"{log_folder}\" \
            --dataidx_filename \"{dataset}/{N}client/{noniid}/{dataset.upper()}-noniid-{noniid}_1.json\" \
            --num_rounds {int(total_epochs/E)} \
            --num_epochs {E} \
            --proportion {K/N:>.1f} \
            --batch_size {batch_size} \
            --num_threads_per_gpu 1  \
            --num_gpus {len(visible_cudas)} \
            --server_gpu_id 0".replace("           ", "")

        file = open(f"./{task_name}_E{E}_{algo}.sh", "w")
        file.write(formated_command)
        file.close()