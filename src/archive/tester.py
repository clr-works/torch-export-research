#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tester.py

Contains the LlamaExportTester class which:
- Runs tests in isolation
- Exports models
- Checks if the model can run inference on exported graphs

Functions & Classes:
    LlamaExportTester
"""

import torch
import warnings
import gc
from contextlib import contextmanager

from test_cases import TestCase
from fms.models import get_model

warnings.filterwarnings("ignore", message="Attempted to insert a get_attr Node")
warnings.filterwarnings("ignore", message="Node.*does not reference")
warnings.filterwarnings("ignore", message="`torch.library.impl_abstract` was renamed")

class LlamaExportTester:
    def __init__(self, model_getter):
        self.model_getter = model_getter
        self.debug_info = {}

    def _reset_state(self):
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @contextmanager
    def _isolated_test_context(self):
        self._reset_state()
        model = None
        try:
            model = self.model_getter()
            model.eval()
            yield model
        finally:
            if model is not None:
                del model
            self._reset_state()

    def run_single_test(self, test_case: TestCase) -> bool:
        with self._isolated_test_context() as model:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                dummy_input = torch.randint(0, 256, test_case.shape, dtype=test_case.dtype).to(device)
                print(f"\nTesting {test_case.name}: {test_case.description}")
                with torch.no_grad():
                    print("- Running pre-export inference...")
                    output = model(dummy_input)
                    print("- Attempting export...")
                    with warnings.catch_warnings():
                        # ignore known warnings
                        warnings.filterwarnings("ignore", message="Attempted to insert a get_attr Node")
                        warnings.filterwarnings("ignore", message="Node.*does not reference")

                        exported_program = torch.export.export(
                            model,
                            (dummy_input,),
                            dynamic_shapes=test_case.dynamic_shapes,
                        )
                    print("- Testing exported model...")
                    with torch.inference_mode():
                        _ = exported_program.module()(dummy_input)
                    print(" Test passed")
                return True
            except Exception as e:
                if test_case.expected_success:
                    print(f" Unexpected failure: {str(e)}")
                else:
                    print(f" Expected failure occurred: {str(e)}")
                return False

    def run_test_suite(self, test_cases):
        results = []
        for test_case in test_cases:
            success = self.run_single_test(test_case)
            results.append({"name": test_case.name, "success": success, "expected": test_case.expected_success})
        return results

    def print_summary(self, results):
        print("\nTest Summary:")
        print("-" * 50)
        passed = 0
        total = len(results)
        for result in results:
            status = "PASS" if result["success"] == result["expected"] else "FAIL"
            if status == "PASS":
                passed += 1
            print(f"{result['name']: <20} {status}")
        print("-" * 50)
        print(f"Passed: {passed}/{total} tests")
