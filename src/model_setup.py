#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_setup.py

Module responsible for:
- Loading the Micro-LLaMA model
- Moving it to the appropriate device
- Putting it into evaluation mode
- Any additional configuration needed before inference/export

Functions:
    load_micro_llama() -> nn.Module
"""

import torch
from fms.models import get_model

def load_micro_llama(model_variant="micro"):
    """
    Load the Micro-LLaMA model variant and move it to the GPU if available.

    Parameters
    ----------
    model_variant : str
        The model variant to load (default: "micro").

    Returns
    -------
    model : torch.nn.Module
        The loaded and prepared Micro-LLaMA model on the appropriate device.
    """
    print(f"Loading Micro-LLaMA model variant: {model_variant}")
    model = get_model("llama", model_variant)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model successfully loaded on {device}")
    return model, device
