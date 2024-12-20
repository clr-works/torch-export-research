# **Model Profiling and Analysis**

This repository provides tools and scripts for profiling, analyzing, and exporting large language models (LLMs) like **LLaMA** and **Granite**. The tools are designed for performance benchmarking and experimentation with model export, inference, and weight injection using PyTorch’s advanced features.

---

## **Features**
- **Export Models**: Export models with static or dynamic sequence lengths.
- **Inject Weights**: Dynamically inject random weights into the model’s state dictionary for testing.
- **Measure Performance**: Record export, injection, and inference times.
- **Graph Analysis**: Analyze graph nodes and operations in exported models.
- **Configuration Management**: Manage model paths and runtime settings using a JSON configuration file.

---

## **Project Structure**
- **`BalancedTParallel.py`**: Script for balanced tensor parallelism.
- **`TensorParallelBluePrint.py`**: Blueprint for setting up tensor parallelism.
- **`analyze_models.py`**: Analyze graph node operations in exported models.
- **`baseline_static_profiling.py`**: Perform static profiling of models for export and inference.
- **`config.json`**: JSON configuration file for managing model paths and parameters.
- **`dynamic_seq_length.py`**: Perform export and inference with dynamic sequence lengths.

---

## **Usage**

### **Injecting Random Weights Dynamically**
This script exports a model, injects random weights, and runs inference. The paths and parameters are managed using `config.json`.

---

### **Steps to Run**
1. **Update `config.json`**:
   Edit the `config.json` file with the paths to the models and default parameters:
   ```json
   {
       "models": {
           "llama-7b": "/path/to/llama-7b",
           "granite-2b": "/path/to/granite-2b"
       },
       "default_model": "granite-2b",
       "samples_to_process": 100
   }

All files except the Tensor files (`BalancedTParallel.py` and `TensorParallelBluePrint.py`) run with the following logic:

```bash
python inject_weights_dynamically.py --model llama-7b --num_samples 100

  

