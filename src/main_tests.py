#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_tests.py

Orchestrates the running of the test suite:
- Gets test cases
- Uses LlamaExportTester
- Prints summary

Run:
    python main_tests.py
"""

import os
import warnings
from tester import LlamaExportTester
from test_cases import create_comprehensive_test_suite
from fms.models import get_model

os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

# Suppress known warnings
warnings.filterwarnings("ignore", message="Attempted to insert a get_attr Node")
warnings.filterwarnings("ignore", message="Node.*does not reference")
warnings.filterwarnings("ignore", message="`torch.library.impl_abstract` was renamed")

if __name__ == "__main__":
    tester = LlamaExportTester(lambda: get_model("llama", "micro"))
    test_cases = create_comprehensive_test_suite()
    results = tester.run_test_suite(test_cases)
    tester.print_summary(results)
