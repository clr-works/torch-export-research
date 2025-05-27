#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plotting.py

Module for plotting inference results:
- Line plots for run-by-run inference times
- Bar charts comparing cold vs warm speeds

Functions:
    plot_inference_times(reg_times, comp_times)
    plot_speedups(cold_speedup, warm_speedup)
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_inference_times(reg_times, comp_times):
    """
    Plot inference times across multiple runs for regular and compiled models.

    Parameters
    ----------
    reg_times : list[float]
        Times for regular model.
    comp_times : list[float]
        Times for compiled model.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(reg_times)+1), reg_times, label="Regular Model", marker="o")
    plt.plot(range(1, len(comp_times)+1), comp_times, label="Compiled Model (AOTI)", marker="s")
    plt.xlabel("Run")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Times Across Runs")
    plt.legend()
    plt.grid()
    plt.savefig("inference_times.png")

def plot_speedups(cold_speedup, warm_speedup):
    """
    Create a bar chart comparing cold and warm inference speedups.

    Parameters
    ----------
    cold_speedup : float
        Speedup for the first (cold) run.
    warm_speedup : float
        Speedup for warm runs (2-10 avg).
    """
    labels = ["Cold Speedup", "Warm Speedup"]
    values = [cold_speedup, warm_speedup]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["skyblue", "lightgreen"])
    plt.ylabel("Speedup (X)")
    plt.title("Comparison of Cold vs Warm Speedups")
    for i, val in enumerate(values):
        plt.text(i, val + 0.05, f"{val:.2f}x", ha='center', va='bottom')
    plt.ylim(0, max(values)*1.2)
    plt.show()
    plt.savefig("speedup_comparison.png")
