# PEFT Fine-tuning of BLOOM-560M Model

This project demonstrates how to fine-tune the BLOOM-560M language model using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically using LoRA (Low-Rank Adaptation). The implementation uses bitsandbytes for 8-bit quantization to reduce memory usage while maintaining model performance.

## Project Overview

The project showcases:
- Loading and quantizing the BLOOM-560M model
- Implementing LoRA adapters for efficient fine-tuning
- Using 8-bit quantization with bitsandbytes
- Training on a text dataset (English quotes)

## Requirements

```
bitsandbytes
datasets
accelerate
loralib
transformers
peft
torch
```

## Setup

1. Install the required packages:
```bash
pip install bitsandbytes datasets accelerate loralib
pip install git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
```

2. Login to Hugging Face Hub (if needed):
```python
from huggingface_hub import notebook_login
notebook_login()
```

## Model Configuration

The project uses the following configurations:

- Base Model: `bigscience/bloomz-560m`
- Loading Configuration:
  - 8-bit quantization enabled
  - Automatic device mapping
  - Gradient checkpointing enabled

- LoRA Configuration:
  - r = 16 (attention heads)
  - lora_alpha = 32
  - lora_dropout = 0.05
  - task_type = "CAUSAL_LM"

## Training Details

The model is fine-tuned using:
- Frozen base model weights
- LoRA adapters for efficient parameter updates
- 8-bit quantization for memory efficiency
- Dataset: Abirate/english_quotes

## Memory Efficiency

The implementation achieves significant memory efficiency:
- Only ~0.28% of parameters are trainable
- Uses 8-bit quantization
- Implements gradient checkpointing

## Usage

The notebook `4_PEFT_Finetune_Bloom_560m_tagger.ipynb` contains the complete implementation and can be run in environments with GPU support.

## Notes

- The model uses automatic device mapping for optimal resource utilization
- Small parameters (e.g., layernorm) are cast to fp32 for stability
- The implementation includes proper handling of mixed precision operations 