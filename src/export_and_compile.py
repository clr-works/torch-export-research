#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_and_compile.py

Module for exporting the model with torch.export and performing AOTInductor compilation.

Functions:
    export_model(model: nn.Module, example_input: torch.Tensor) -> torch.export.ExportedProgram
    compile_with_aot(exported_program, example_input, output_path: str, options: dict) -> str
"""

import torch
from torch.export import export, save
from torch._inductor import aot_compile

def export_model(model, example_input, export_path="llama_micro_exported.pt2"):
    """
    Export the given model using torch.export with the specified example input.

    Parameters
    ----------
    model : torch.nn.Module
        The model to export.
    example_input : torch.Tensor
        A representative input for tracing/export.
    export_path : str
        File path to save the exported program.

    Returns
    -------
    exported_program : torch.export.ExportedProgram
        The exported program object.
    """
    print("Exporting the model...")
    with torch.no_grad():
        exported_program = export(model, (example_input,))
    save(exported_program, export_path)
    print(f"Model successfully exported and saved as '{export_path}'")
    return exported_program

def compile_with_aot(exported_program, example_input, so_path="llama_micro_compiled.so", compile_options=None):
    """
    AOT compile the exported program using AOTInductor.

    Parameters
    ----------
    exported_program : torch.export.ExportedProgram
        The exported program to compile.
    example_input : torch.Tensor
        Example input tensor to guide compilation.
    so_path : str
        Where to save the compiled shared object.
    compile_options : dict
        AOT compile options.

    Returns
    -------
    compiled_so : str
        Path to the compiled shared object file.
    """
    if compile_options is None:
        compile_options = {"aot_inductor.output_path": so_path}
    with torch.no_grad():
        print("Compiling the model with AOTInductor...")
        compiled_so = aot_compile(exported_program.module(), (example_input,), options=compile_options)
        print(f"Model compiled successfully and saved at {so_path}")
    return compiled_so
