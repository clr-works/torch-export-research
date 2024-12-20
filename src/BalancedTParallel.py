import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import datetime

'''this code implements a balanced approach to tensor parallelism to avoid bottlenecks we apply allgather'''

def debug_print(msg):
    rank = os.environ.get('LOCAL_RANK', 'Not initialized')
    print(f"[Rank {rank}] {msg}", flush=True)

def init_process_group():
    try:
        debug_print("CUDA Available: " + str(torch.cuda.is_available()))
        debug_print("GPU Count: " + str(torch.cuda.device_count()))
        debug_print("NCCL Available: " + str(dist.is_nccl_available()))
        
        local_rank = int(os.environ['LOCAL_RANK'])
        debug_print(f"Local rank: {local_rank}")
        
        backend = 'nccl' if torch.cuda.is_available() and dist.is_nccl_available() else 'gloo'
        debug_print(f"Using {backend} backend")
        
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            timeout=datetime.timedelta(seconds=60)
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            debug_print(f"Set CUDA device to: {torch.cuda.current_device()}")
        
        return local_rank
    except Exception as e:
        debug_print(f"Error in init_process_group: {str(e)}")
        raise

class BalancedTensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_gpus=4):
        super().__init__()
        self.num_gpus = num_gpus
        self.out_features_per_gpu = out_features // num_gpus
        self.in_features_per_gpu = in_features // num_gpus
        
        local_rank = dist.get_rank()
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        
        self.linear = nn.Linear(self.in_features_per_gpu, self.out_features_per_gpu).to(device)
        self.activation = nn.ReLU().to(device)
    
    @staticmethod
    def all_gather_outputs(local_output, world_size, device):
        local_size = local_output.size()
        
        # create a list of tensors for gathering
        gathered_list = []
        
        # perform  the all_gather
        for _ in range(world_size):
            gathered_list.append(torch.zeros_like(local_output, device=device))
        
        # Create a new tensor for gathering
        dist.all_gather(gathered_list, local_output.detach())
        
        # Return concatenated result
        return torch.cat(gathered_list, dim=-1)
    
    def forward(self, x):
        local_rank = dist.get_rank()
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        world_size = dist.get_world_size()
        
        # Splitting input
        x_splits = x.chunk(self.num_gpus, dim=-1)
        x_local = x_splits[local_rank].to(device)
        
        # local portion part of prcessing
        partial_out = self.activation(self.linear(x_local))
        
        # gather and concat
        return self.all_gather_outputs(partial_out, world_size, device)

class BalancedParallelNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=2048, output_size=512, num_gpus=4):
        super().__init__()
        self.layer1 = BalancedTensorParallelLinear(input_size, hidden_size, num_gpus)
        self.layer2 = BalancedTensorParallelLinear(hidden_size, output_size, num_gpus)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    try:
        # Environment vars
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        debug_print("Starting script")
        local_rank = init_process_group()
        
        # Create model and input
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        model = BalancedParallelNetwork()
        example_input = torch.randn(32, 512, device=device)
        
        model.eval()
        with torch.no_grad():
            exported_program = torch.export.export(
                model,
                (example_input,)
            )
            
            if dist.get_rank() == 0:
                #exported_program.save("balanced_parallel_model.pt")
                debug_print("Model exported successfully!")
                print(exported_program)
        
        dist.barrier()
        dist.destroy_process_group()
        debug_print("Clean up complete")
        
    except Exception as e:
        debug_print(f"Error in main: {str(e)}")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
