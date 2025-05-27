#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_benchmark.py

The main script to:
- Load the model
- Perform dummy inference
- Export the model
- AOT compile the model
- Run inference and measure speeds
- Verify consistency
- Plot results (including bar charts)

This script uses other modules for modularization and clarity.
"""

import warnings
import torch
import os
import numpy as np

from model_setup import load_micro_llama
from export_and_compile import export_model, compile_with_aot
from inference_utils import measure_inference_times, verify_output
from plotting import plot_inference_times, plot_speedups

# Suppress known warnings
warnings.filterwarnings("ignore", message="Attempted to insert a get_attr Node")
warnings.filterwarnings("ignore", message="Node.*does not reference")
warnings.filterwarnings("ignore", message="`torch.library.impl_abstract` was renamed")

# Ignore all numpy errors
np.seterr(all='ignore')

# Set environment variables for torch logging
os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_VERBOSE"] = "0"


if __name__ == "__main__":
    model, device = load_micro_llama("micro")

    # Prepare dummy input
    dummy_input = torch.randint(0, 256, (1, 10), dtype=torch.long).to(device)

    # Run a dummy inference to ensure correctness
    with torch.no_grad():
        print("Running initial inference on dummy input...")
        output = model(dummy_input)
        print(f"Inference successful! Output shape: {output.shape}")

    # Export
    exported_program = export_model(model, dummy_input, export_path="llama_micro_exported.pt2")

    # Compile
    so_path = os.path.join(os.getcwd(), "llama_micro_compiled.so")
    compile_with_aot(exported_program, dummy_input, so_path=so_path)

    # Reload original model and do a fresh run
    model, device = load_micro_llama("micro")
    dummy_input = torch.randint(0, 256, (1, 10), dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Export and compile again (re-running to ensure no graph breaks)
    exported_program = export_model(model, dummy_input, export_path="llama_micro_exported.pt2")
    compile_with_aot(exported_program, dummy_input, so_path=so_path)

    # Load compiled model
    compiled_model = torch._export.aot_load(so_path, device=device)
    with torch.inference_mode():
        compiled_output = compiled_model(dummy_input)

    # Verify consistency
    max_diff = verify_output(output, compiled_output)
    print(f"Maximum difference between outputs: {max_diff}")

    # Measure inference times
    print("Measuring inference times over 10 runs...")
    reg_times = measure_inference_times(model, dummy_input, runs=10) # Regular Model
    comp_times = measure_inference_times(compiled_model, dummy_input, runs=10) # AOTI Model

    # Statistics
    reg_first = reg_times[0]
    reg_warm_avg = np.mean(reg_times[1:])
    comp_first = comp_times[0]
    comp_warm_avg = np.mean(comp_times[1:])

    cold_speedup = reg_first / comp_first
    warm_speedup = reg_warm_avg / comp_warm_avg

    print(f"Regular Model - First Inference: {reg_first:.6f}s, Warm Avg (2-10): {reg_warm_avg:.6f}s")
    print(f"Compiled Model - First Inference: {comp_first:.6f}s, Warm Avg (2-10): {comp_warm_avg:.6f}s")
    print(f"Cold Speedup: {cold_speedup:.2f}x, Warm Speedup: {warm_speedup:.2f}x")

    # Plots
    plot_inference_times(reg_times, comp_times)
    plot_speedups(cold_speedup, warm_speedup)
