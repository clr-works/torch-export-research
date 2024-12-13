import transformers
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fms.models import get_model
from fms.models.hf import to_hf_api

# Load model directly

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b")

'''
# Paths to your LLaMA 7B model and tokenizer
model_path = "/home/corina_rios/.cache/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba"
#model_path = "/home/corina/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

# Configuration for the model
architecture = "llama"  # Specify the architecture
variant = "13b"  # Specify the variant (e.g., 7B for the smallest model)

torch.set_default_device("cuda")
torch.set_default_dtype(torch.half)

model = get_model(architecture, variant, model_path=model_path, source="hf", device_type="cuda", norm_eps=1e-6)
model = to_hf_api(model)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Use Hugging Face's AutoTokenizer

pipe = pipeline(task="text-generation", model=model, max_new_tokens=25, tokenizer=tokenizer, device="cuda")
prompt = """I believe the meaning of life is"""
result = pipe(prompt)
print(result)'''