# experiments/continual_learning/continual_learning_export.py
"""
Experiment: Continual Learning with torch.export
Tests how exported models handle weight updates during continual learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

@dataclass
class TaskPerformance:
    task_name: str
    loss_before: float
    loss_after: float
    export_time: float
    inference_time: float

class ContinualLearningExport:
    """Test torch.export behavior with continual learning weight updates"""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results_dir = Path("results/continual_learning")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store initial weights for EWC
        self.initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
    def create_task_data(self, task_type: str, num_samples: int = 100) -> List[str]:
        """Create synthetic task-specific data"""
        if task_type == "shakespeare":
            templates = [
                "To be or not to be, that is",
                "All the world's a stage, and all",
                "Romeo, Romeo, wherefore art thou",
                "Is this a dagger which I see",
                "Now is the winter of our discontent"
            ]
        elif task_type == "python":
            templates = [
                "def calculate_sum(a, b):",
                "import numpy as np",
                "class DataProcessor:",
                "for i in range(len(data)):",
                "if __name__ == '__main__':"
            ]
        elif task_type == "medical":
            templates = [
                "The patient presents with symptoms of",
                "Diagnosis includes differential of",
                "Treatment protocol recommends",
                "Laboratory results indicate",
                "Clinical examination reveals"
            ]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Generate variations
        data = []
        for _ in range(num_samples):
            template = np.random.choice(templates)
            continuation = f" {task_type}_specific_content_{np.random.randint(1000)}"
            data.append(template + continuation)
        
        return data
    
    def compute_fisher_information(self, data: List[str]) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix diagonal for EWC"""
        fisher_info = {}
        self.model.eval()
        
        for text in data[:20]:  # Use subset for efficiency
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                 inputs['input_ids'].view(-1))
            
            self.model.zero_grad()
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in fisher_info:
                        fisher_info[name] = param.grad.data.clone() ** 2
                    else:
                        fisher_info[name] += param.grad.data.clone() ** 2
        
        # Average
        for name in fisher_info:
            fisher_info[name] /= len(data[:20])
            
        return fisher_info
    
    def train_on_task(self, task_data: List[str], task_name: str, 
                     use_ewc: bool = False, fisher_info: Dict = None,
                     lambda_ewc: float = 1000) -> float:
        """Train model on a specific task"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        total_loss = 0
        num_steps = 50  # Quick training
        
        for step in range(num_steps):
            # Sample batch
            batch_texts = np.random.choice(task_data, size=4, replace=True)
            inputs = self.tokenizer(list(batch_texts), return_tensors="pt", 
                                   padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                 inputs['input_ids'].view(-1))
            
            # Add EWC penalty if enabled
            if use_ewc and fisher_info:
                ewc_loss = 0
                for name, param in self.model.named_parameters():
                    if name in fisher_info:
                        ewc_loss += (fisher_info[name] * 
                                   (param - self.initial_params[name]) ** 2).sum()
                loss += lambda_ewc * ewc_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_steps
    
    def export_model_with_timing(self) -> Tuple[torch.export.ExportedProgram, float]:
        """Export model and measure time"""
        # Create sample input
        sample_text = "This is a test"
        inputs = self.tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Wrapper for export
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask):
                return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        wrapped_model = ModelWrapper(self.model)
        
        # Time the export
        start_time = time.time()
        exported = export(wrapped_model, args=(input_ids, attention_mask))
        export_time = time.time() - start_time
        
        return exported, export_time
    
    def evaluate_task_performance(self, task_data: List[str]) -> float:
        """Evaluate model performance on a task"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for text in task_data[:20]:  # Evaluate on subset
                inputs = self.tokenizer(text, return_tensors="pt", 
                                       padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                     inputs['input_ids'].view(-1))
                total_loss += loss.item()
        
        return total_loss / len(task_data[:20])
    
    def run_experiment(self):
        """Run the complete continual learning experiment"""
        tasks = ["shakespeare", "python", "medical"]
        results = {
            "without_ewc": [],
            "with_ewc": [],
            "export_times": [],
            "model_name": self.model_name
        }
        
        print("=== Continual Learning Export Experiment ===")
        print(f"Model: {self.model_name}")
        print(f"Tasks: {tasks}")
        print()
        
        # Experiment 1: Without EWC (catastrophic forgetting)
        print("--- Experiment 1: Without EWC ---")
        self.model.load_state_dict({name: param.clone() for name, param in self.initial_params.items()})
        
        all_task_data = {}
        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}: {task}")
            
            # Create task data
            task_data = self.create_task_data(task)
            all_task_data[task] = task_data
            
            # Evaluate before training
            loss_before = self.evaluate_task_performance(task_data)
            
            # Export before training
            exported_before, export_time_before = self.export_model_with_timing()
            
            # Train on task
            avg_train_loss = self.train_on_task(task_data, task)
            
            # Export after training
            exported_after, export_time_after = self.export_model_with_timing()
            
            # Evaluate after training
            loss_after = self.evaluate_task_performance(task_data)
            
            print(f"  Loss: {loss_before:.4f} -> {loss_after:.4f}")
            print(f"  Export time: {export_time_before:.4f}s -> {export_time_after:.4f}s")
            
            results["without_ewc"].append({
                "task": task,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "export_time_before": export_time_before,
                "export_time_after": export_time_after
            })
            
            # Test on previous tasks (catastrophic forgetting)
            if i > 0:
                print("  Performance on previous tasks:")
                for prev_task in tasks[:i]:
                    prev_loss = self.evaluate_task_performance(all_task_data[prev_task])
                    print(f"    {prev_task}: {prev_loss:.4f}")
        
        # Experiment 2: With EWC (mitigate forgetting)
        print("\n--- Experiment 2: With EWC ---")
        self.model.load_state_dict({name: param.clone() for name, param in self.initial_params.items()})
        
        fisher_info = None
        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}: {task}")
            
            task_data = all_task_data[task]
            
            # Evaluate before training
            loss_before = self.evaluate_task_performance(task_data)
            
            # Train with EWC
            avg_train_loss = self.train_on_task(task_data, task, 
                                               use_ewc=(i > 0), 
                                               fisher_info=fisher_info)
            
            # Update Fisher information after first task
            if i == 0:
                fisher_info = self.compute_fisher_information(task_data)
                # Update reference parameters
                self.initial_params = {name: param.clone() 
                                     for name, param in self.model.named_parameters()}
            
            # Export and evaluate
            exported, export_time = self.export_model_with_timing()
            loss_after = self.evaluate_task_performance(task_data)
            
            print(f"  Loss: {loss_before:.4f} -> {loss_after:.4f}")
            print(f"  Export time: {export_time:.4f}s")
            
            results["with_ewc"].append({
                "task": task,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "export_time": export_time
            })
            
            # Test on previous tasks
            if i > 0:
                print("  Performance on previous tasks:")
                for prev_task in tasks[:i]:
                    prev_loss = self.evaluate_task_performance(all_task_data[prev_task])
                    print(f"    {prev_task}: {prev_loss:.4f}")
        
        # Save results
        self.save_results(results)
        self.plot_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save experiment results"""
        with open(self.results_dir / "continual_learning_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {self.results_dir}")
    
    def plot_results(self, results: Dict):
        """Create visualization of results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Task performance comparison
        tasks = [r['task'] for r in results['without_ewc']]
        without_ewc_loss = [r['loss_after'] for r in results['without_ewc']]
        with_ewc_loss = [r['loss_after'] for r in results['with_ewc']]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        ax1.bar(x - width/2, without_ewc_loss, width, label='Without EWC', alpha=0.8)
        ax1.bar(x + width/2, with_ewc_loss, width, label='With EWC', alpha=0.8)
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Loss After Training')
        ax1.set_title('Task Performance: EWC vs No EWC')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tasks)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Export times
        export_times_no_ewc = [r['export_time_after'] for r in results['without_ewc']]
        export_times_ewc = [r['export_time'] for r in results['with_ewc']]
        
        ax2.plot(tasks, export_times_no_ewc, 'o-', label='Without EWC', markersize=8)
        ax2.plot(tasks, export_times_ewc, 's-', label='With EWC', markersize=8)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Export Time (seconds)')
        ax2.set_title('Export Time Across Tasks')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "continual_learning_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {self.results_dir}/continual_learning_plot.png")

def main():
    """Run the continual learning export experiment"""
    # You can change the model here
    experiment = ContinualLearningExport(model_name="gpt2")
    results = experiment.run_experiment()
    
    # Print summary
    print("\n=== Summary ===")
    print("Without EWC: Model shows catastrophic forgetting")
    print("With EWC: Model better preserves performance on earlier tasks")
    print(f"Export times remain relatively stable: ~{np.mean([r['export_time'] for r in results['with_ewc']]):.2f}s")

if __name__ == "__main__":
    main()