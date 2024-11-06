# IBM_pr3

# Project Proposal 3: Pre-compiling Models Using `torch.export()` + AOTInductor

**Supervisor:** Dr. Antoni Viros i Martin

---

## Abstract
This project explores the use of `torch.export()` and AOTInductor to pre-compile models for faster inference. The goal is to test whether model weights can be separated from runtime in compiled models, thereby improving the deployment flexibility of IBM Foundation Model Stack (FMS) models. Although research-focused, separating weights and runtime could reveal valuable insights for AIU applications. Regular GPUs will be sufficient for experimentation.

---

## Objectives
1. Use `torch.export()` and AOTInductor to pre-compile FMS models.
2. Experiment with separating weights and runtime in the compiled models.
3. Benchmark inference performance on regular GPUs.

---

## Tools and Resources
- **IBM FMS:** [IBM FMS on Hugging Face](https://huggingface.co/ibm-fms)
- **Libraries:** PyTorch `torch.export()` and AOTInductor

---

## Project Timeline (8 Weeks)

- **Weeks 1-2:** Explore `torch.export()` and AOTInductor capabilities.
- **Weeks 3-5:** Implement model pre-compilation workflows with FMS models.
- **Weeks 6-7:** Test separating model weights and runtime.
- **Week 8:** Document findings and provide an analysis of inference performance.

--- 
