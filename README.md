Thank you for the clarification. Here's the updated README with the additional information you provided:

# **Model Profiling and Analysis**

This repository provides tools and scripts for profiling, analyzing, and exporting large language models (LLMs) like **LLaMA** and **Granite**. The tools are designed for performance benchmarking and experimentation with model export, inference, and weight injection using PyTorch's advanced features.

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
This script exports a model, injects random weights, and runs inference. The paths and parameters are managed using `config.json`.

---

### **Steps to Run**
1. **Update `config.json`**:
   Edit the `config.json` file with the paths to your models and default parameters:
   ```json
   {
       "models": {
           "llama-7b": "/path/to/llama-7b",
           "granite-2b": "/path/to/granite-2b"
       },
       "default_model": "granite-2b",
       "samples_to_process": 100
   }
   ```

All files except the Tensor files (`BalancedTParallel.py` and `TensorParallelBluePrint.py`) run with the following logic:
```bash
python inject_weights_dynamically.py --model llama-7b --num_samples 100
```

The scripts in this repository provide various functionalities:
- `analyze_models.py`: Analyzes the graph node operations in exported models.
- `baseline_static_profiling.py`: Performs static profiling of models for export and inference.
- `dynamic_seq_length.py`: Performs export and inference with dynamic sequence lengths.
- `export_and_compile.py`: Exports and compiles models.
- `inference_utils.py`: Provides utility functions for performing inference.
- `inject_weights_dynamically.py`: Injects random weights into the model dynamically.
- `main_benchmark.py`: Benchmarks the torch.export of micro LLaMA and evaluates compiled vs uncompiled models.
- `main_tests.py`: Runs tests to evaluate graph breaks.
- `model_setup.py`: Provides setup and configuration for the models.
- `plotting.py`: Contains functions for plotting the results.
- `test_cases.py`: Defines test cases for the models.
- `tester.py`: Implements a tester class for running tests on the models.