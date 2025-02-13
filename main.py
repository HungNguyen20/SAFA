import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing
import wandb
import time


def wandb_log(input, name, step):
    for (i, v) in enumerate(input):
        wandb.log({f"{name}/{name}-{i}": v}, step)
    return


class MyLogger(flw.Logger):
    def __init__(self):
        super().__init__()
        self.max_acc = 0
        
    def log(self, server=None, round=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "mean_curve":[],
                "var_curve":[],
                "test_accs":[],
                "test_losses":[],
                "valid_accs":[],
                "client_accs":{},
                "mean_valid_accs":[],
                "max_acc": [],
                "offset": []
            }
        
        with torch.no_grad():
            self.time_start('Global Testing Time')
            test_metric, test_loss = server.test(device='cuda')
            self.time_end('Global Testing Time')
        
        self.max_acc = max(self.max_acc, test_metric)
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        self.output['max_acc'].append(self.max_acc)
        
        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name]=[self.output['valid_accs'][i][cid] for i in range(len(self.output['valid_accs']))]
        
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
        print(self.temp.format("Max of Global Accuracy:", self.max_acc))
        
        try:
            self.output['offset'].append(server.offset)
            print(self.temp.format("Offset:", int(server.offset)))
        except:
            self.output['offset'].append(1)

        # wandb record
        if server.wandb:
            wandb.log(
                {
                    "Testing Loss":         self.output['test_losses'][-1],
                    "Testing Accuracy":     self.output['test_accs'][-1],
                    "Max Testing Accuracy": self.max_acc,
                },
                step=round
            )

            try:
                # Log gain scale
                wandb.log({"Offset": server.offset,}, step=round)
                wandb_log(server.gain_tracking.tolist(), "Singular Gain", step=round)
                wandb_log(server.forget_tracking.tolist(), "Singular Forget", step=round)
            except:
                pass


logger = MyLogger()

def main():
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    
    runname = f"{option['algorithm']}_"
    for para in server.paras_name:
        runname = runname + para + "{}_".format(option[para])
    
    if option['wandb']:
        wandb.init(
            project="project-name", 
            entity="entity-name",
            group=option['task'],
            name=runname[:-1],
            config=option
        )
    
    print("CONFIG =>", option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




