#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference_utils.py

Module providing utilities for running inference and measuring performance.

Functions:
    measure_inference_times(model: nn.Module, input_tensor: torch.Tensor, runs: int) -> List[float]
    verify_output(output1: torch.Tensor, output2: torch.Tensor) -> float
"""

import time
import torch
import numpy as np
from typing import List

def measure_inference_times(model, input_tensor, runs=10):
    """
    Measure inference times over multiple runs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run inference on.
    input_tensor : torch.Tensor
        The input for inference.
    runs : int
        Number of runs to measure.

    Returns
    -------
    times : List[float]
        A list of inference times for each run.
    """
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start_time = time.time()
            _ = model(input_tensor)
            times.append(time.time() - start_time)
    return times

def verify_output(output1, output2):
    """
    Compute the maximum difference between two model outputs.

    Parameters
    ----------
    output1 : torch.Tensor
    output2 : torch.Tensor

    Returns
    -------
    max_difference : float
        The maximum absolute difference between outputs.
    """
    max_difference = torch.max(torch.abs(output1 - output2)).item()
    return max_difference
