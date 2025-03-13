# **Profiling and Analysis**

This project explores the use of torch.export() and AOTInductor to pre-compile models for
faster inference. The goal is to test whether model weights can be separated from runtime
in compiled models, improving the deployment flexibility of IBM Foundation Model Stack
(FMS) models. Although the project is research-focused, the separation of weights and
runtime could unlock valuable insights for AIU applications. 

---

## **Features**
- **Export Models**: Export models with static or dynamic sequence lengths.
- **Inject Weights**: Dynamically inject random weights into the model's state dictionary for testing.
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
- **`export_and_compile.py`**: Export and compile models.
- **`inference_utils.py`**: Utility functions for performing inference.
- **`inject_weights_dynamically.py`**: Inject random weights into the model dynamically.
- **`main_benchmark.py`**: Main script for benchmarking the torch.export of micro LLaMA and evaluating compiled vs uncompiled models.
- **`main_tests.py`**: Main script for running tests to evaluate graph breaks.
- **`model_setup.py`**: Setup and configuration for models.
- **`plotting.py`**: Functions for plotting results.
- **`test_cases.py`**: Test cases for the models.
- **`tester.py`**: Tester class for running tests.

---

## **Usage**

### **Running Tests and Benchmarks**
To run the main tests for evaluating graph breaks, use the following command:
```bash
python3 main_tests.py
```

To run the main benchmark for torch.export of micro LLaMA and evaluation of compiled vs uncompiled models, use the following command:
```bash
python3 main_benchmark.py
```

### **Injecting Random Weights Dynamically**
This script exports a model, injects random weights, and runs inference. The paths are managed using `config.json`.

---

### **Steps to Run**
1. **Update `config.json`**:
   Edit the `config.json` file with the paths to the models and default parameters:
   ```json
   {
       "models": {
           "llama-7b": "/path/to/llama-7b",
           "granite-2b": "/path/to/granite-2b"
       }
       

All files except the Tensor files (`BalancedTParallel.py` and `TensorParallelBluePrint.py`) run with the following logic:
```bash
python inject_weights_dynamically.py --model llama-7b --num_samples 100
```

## Results and Artifacts
In the resuls folder there are .csv files with all experiments that have been run. Additionally, some visualizations. The .so file with Micro llama compiled and the exported Micro llama are stored to showcase.
