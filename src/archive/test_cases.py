#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_cases.py

Defines the TestCase dataclass and creates a comprehensive suite of test cases
for LLaMA model export testing.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import torch

@dataclass
class TestCase:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    expected_success: bool
    dynamic_shapes: Optional[Dict[str, Any]] = None
    description: str = ""

def create_comprehensive_test_suite() -> List[TestCase]:
    """Creates a comprehensive test suite for LLaMA model export testing."""
    # Basic functionality tests
    basic_tests = [
        TestCase("static_tiny", (1, 8), torch.long, True, description="Minimal working input size"),
        TestCase("static_small", (1, 16), torch.long, True, description="Small but typical input size"),
        TestCase("static_medium", (1, 32), torch.long, True, description="Medium input size"),
    ]

    # Sequence length tests
    sequence_tests = [
        TestCase(f"seq_len_{length}", (1, length), torch.long, True, description=f"Testing sequence length {length}")
        for length in [64, 128, 256, 512, 1024, 2048]
    ]

    # Batch size tests
    batch_tests = [
        TestCase(f"batch_{size}", (size, 32), torch.long, True, description=f"Testing batch size {size}")
        for size in [2, 4, 8, 16, 32, 64, 128]
    ]

    # Dynamic shape tests
    dynamic_tests = [
        TestCase("dynamic_batch", (4, 32), torch.long, True,
                 dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}},
                 description="Dynamic batch dimension"),
        TestCase("dynamic_seq", (1, 64), torch.long, True,
                 dynamic_shapes={"x": {1: torch.export.Dim("seq", min=32, max=128)}},
                 description="Dynamic sequence length"),
        TestCase("dynamic_both", (4, 64), torch.long, True,
                 dynamic_shapes={
                     "x": {
                         0: torch.export.Dim("batch", min=1, max=8),
                         1: torch.export.Dim("seq", min=32, max=128)
                     }
                 },
                 description="Dynamic batch and sequence length"),
    ]

    # Edge cases
    edge_cases = [
        TestCase("zero_batch", (0, 32), torch.long, False, description="Zero batch size should fail"),
        TestCase("zero_seq", (1, 0), torch.long, False, description="Zero sequence length - should fail"),
        TestCase("single_token", (1, 1), torch.long, True, description="Single token input"),
        TestCase("large_batch_small_seq", (256, 4), torch.long, True, description="Large batch with small sequence"),
    ]

    # Data type tests
    dtype_tests = [
        TestCase("wrong_dtype_float", (1, 32), torch.float32, False, description="Incorrect dtype float32"),
        TestCase("wrong_dtype_int", (1, 32), torch.int32, False, description="Incorrect dtype int32"),
    ]

    # Memory boundary tests
    memory_tests = [
        TestCase("memory_intensive", (64, 2048), torch.long, True, description="Memory intensive configuration"),
    ]

    # Special sequence length tests for positional encoding
    position_tests = [
        TestCase(f"position_boundary_{length}", (1, length), torch.long, True,
                 description=f"Testing near positional encoding boundary {length}")
        for length in [1020, 1022, 1023]
    ]

    # Combined variation tests
    combined_tests = [
        TestCase("mixed_large_batch_seq", (32, 512), torch.long, True,
                 description="Moderate batch and sequence size"),
        TestCase("mixed_dynamic", (8, 256), torch.long, True,
                 dynamic_shapes={
                     "x": {
                         0: torch.export.Dim("batch", min=4, max=16),
                         1: torch.export.Dim("seq", min=128, max=512)
                     }
                 },
                 description="Dynamic dimensions with moderate sizes"),
    ]

    all_tests = (
        basic_tests +
        sequence_tests +
        batch_tests +
        dynamic_tests +
        edge_cases +
        dtype_tests +
        memory_tests +
        position_tests +
        combined_tests
    )
    return all_tests
