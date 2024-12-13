import torch
import torch.nn as nn
import torch.distributed as dist

'''this code implements a the basic blueprint for a multi gpu environment exporting using tensor parallelism. 
GPU 0 acts as the collector for concatenating outputs and with a large model can potentially be a bottleneck '''

class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_gpus=4):
        super().__init__()
        
        self.num_gpus = num_gpus
        # split the output features across GPUs
        assert out_features % num_gpus == 0, "out_features must be divisible by num_gpus"
        self.out_features_per_gpu = out_features // num_gpus
        
        # create separate linear layers for each GPU
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features, self.out_features_per_gpu)
            for _ in range(num_gpus)
        ])
        
        # moving layer to respectivge GPUs
        for gpu_id, layer in enumerate(self.linear_layers):
            layer.to(f'cuda:{gpu_id}')
    
    def forward(self, x):
        partial_outputs = []
        
        # Processng input on each GPU
        for gpu_id, layer in enumerate(self.linear_layers):
            # move input chunk to current GPU
            x_gpu = x.to(f'cuda:{gpu_id}')
            # compute partial output
            partial_out = layer(x_gpu)
            partial_outputs.append(partial_out)
        
        # Concatenate results from all GPUs, bottleneck
        # Move all partial outputs to GPU 0 for concatenation
        partial_outputs = [out.to('cuda:0') for out in partial_outputs]
        return torch.cat(partial_outputs, dim=-1)

class SimpleParallelNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=2048, output_size=512, num_gpus=4):
        super().__init__()
        
        self.num_gpus = num_gpus
        
        #first layer: tensor parallel linear transformation
        self.layer1 = TensorParallelLinear(input_size, hidden_size, num_gpus)
        
        #activation function (runs on GPU 0)
        self.activation = nn.ReLU()
        
        # second layer: tensor parallel linear transformation
        self.layer2 = TensorParallelLinear(hidden_size, output_size, num_gpus)
    
    def forward(self, x):
        # input starts on GPU 0
        x = x.to('cuda:0')
        
        x = self.layer1(x)
        
        x = self.activation(x)
        
        x = self.layer2(x)
        
        return x

def export_parallel_model():
    # Initialize model
    model = SimpleParallelNetwork(
        input_size=512,
        hidden_size=2048,
        output_size=512,
        num_gpus=4
    )
    
    # example input
    example_input = torch.randn(32, 512)
    
    model.eval()  # Set to evaluation mode
    exported_program = torch.export.export(
        model,
        (example_input,)
    )
    
    # Save the exported model
    #save("tensor_parallel_model.pt")
    return exported_program

if __name__ == "__main__":
    #check four GPUs availability
    assert torch.cuda.device_count() >= 4,
    
    exported_program = export_parallel_model()
    print("Model exported successfully!")