# Torch Export Research: Decoupling Weights from Runtime

A profiling toolkit for investigating weight-runtime decoupling in PyTorch models to enable faster deployment through weight injection.

## 🎯 Motivation

Traditional model deployment couples weights with computational graphs, requiring full model re-export when weights change. This research explores decoupling these components to enable:
- **Faster deployment cycles** - Update weights without re-exporting
- **Reduced export overhead** - One-time export, multiple weight updates
- **Dynamic experimentation** - Hot-swap weights for A/B testing

## 🔍 Key Findings

Our profiling reveals:

1. **Export Time Scaling**: Significant overhead that increases with model size
   - GPT-2 (124M): ~2s export time
   - LLaMA 7B: ~45s export time

2. **Runtime Performance**: Relative differences between models are smaller during inference/injection than export
   - Suggests export is the bottleneck, not runtime weight injection
   - Larger models show less pronounced performance gaps during actual execution

3. **Dynamic Shapes**: Successfully resolved torch.export constraints by:
   - Using tuple-based input wrappers
   - Constraining sequence lengths to 8*n multiples
   - Enabling variable-length input support

## 🛠️ Project Structure

```
torch-export-research/
├── src/                          # Core profiling tools
│   ├── benchmarks/              
│   │   ├── static_export.py     # Traditional export baseline
│   │   ├── dynamic_export.py    # Dynamic shape support
│   │   └── weight_injection.py  # Weight decoupling tests
│   ├── profiling/               # Performance analysis
│   └── utils/                   # Shared utilities
├── experiments/                  # Proof of concepts
│   ├── continual_learning/      # Weight updates during learning
│   ├── model_surgery/           # Runtime architecture changes  
│   └── distributed_weights/     # Weight streaming scenarios
├── results/                      # Benchmark outputs
└── configs/                      # Model configurations
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/torch-export-research
cd torch-export-research
pip install -r requirements.txt
```

### Run Benchmarks
```bash
# Single model benchmark
python benchmark_runner.py --model gpt2 --benchmark weight_injection

# Full experiment suite
./run_experiments.sh
```

### Configuration
Models are configured in `configs/config.json`:
```json
{
    "gpt2": "gpt2",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-7b": "/path/to/llama-7b"
}
```

## 📊 Benchmark Types

### 1. **Static Export** (Baseline)
Traditional torch.export with fixed shapes
```bash
python benchmark_runner.py --model gpt2 --benchmark static
```

### 2. **Dynamic Export** 
Export with variable sequence lengths
```bash
python benchmark_runner.py --model gpt2 --benchmark dynamic \
    --test-seq-lengths "32,128,512"
```

### 3. **Weight Injection**
Decouple and inject weights post-export
```bash
python benchmark_runner.py --model gpt2 --benchmark weight_injection
```

## 🧪 Proof of Concepts

### Continual Learning
Tests weight updates during sequential task learning:
```bash
python experiments/continual_learning/continual_learning_export.py
```
- Demonstrates export stability with evolving weights
- Compares catastrophic forgetting vs. EWC protection

### Model Surgery (Planned)
- Runtime layer replacement
- Architecture morphing without re-export

### Distributed Weights (Planned)
- Streaming weights from remote storage
- Partial weight updates for large models

## 📈 Results

Benchmark results are saved as CSVs with:
- Export times
- Inference latencies  
- Weight injection overhead
- Memory usage patterns

Visualizations available in Jupyter notebooks under `notebooks/`

## 🔧 Extending the Framework

### Add New Benchmark
```python
from benchmark_framework import BenchmarkBase

class MyBenchmark(BenchmarkBase):
    def run_experiment(self, sample_input):
        # Your profiling logic
        pass
```

### Add New Experiment
```python
from experiments.base_experiment import ExperimentBase

class MyExperiment(ExperimentBase):
    def run(self):
        # Your proof of concept
        pass
```

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional weight injection strategies
- Optimizations for specific model architectures
- New proof-of-concept experiments
- Performance improvements

## 📚 Citation

If you use this work in your research:
```bibtex
@software{torch_export_research,
  title={Torch Export Research: Decoupling Weights from Runtime},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/torch-export-research}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for torch.export functionality
- Hugging Face for model implementations
- Contributors and testers

---

**Note**: This is active research. APIs and approaches may change as we explore optimal weight decoupling strategies.
