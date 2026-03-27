# 🦥 LLM Fine-Tuning with Unsloth

> A beginner-to-intermediate guide for fine-tuning large language models faster and with less memory.

---

## 📌 What is Unsloth?

[Unsloth](https://github.com/unslothai/unsloth) is an open-source Python library that makes LLM fine-tuning **~2x faster** and uses **~60% less VRAM** compared to standard HuggingFace training — without any accuracy loss.

---

## 📓 Notebook Overview

This notebook walks through the full fine-tuning pipeline step by step:

| Section | Topic |
|---------|-------|
| 1 | Installation & Library Imports |
| 2 | Loading the Model (`FastLanguageModel`) |
| 3 | Adding a LoRA Adapter (PEFT) |
| 4 | Dataset Preparation & Prompt Template |
| 5 | Training Configuration with `SFTTrainer` |
| 6 | Inference — Testing the Fine-Tuned Model |
| 7 | Saving the Model & GGUF Export |
| 8 | Bonus: Unsloth vs Standard Comparison Table |

---

## ⚡ Key Concepts

### LoRA (Low-Rank Adaptation)
Instead of updating all billions of parameters, LoRA inserts small trainable adapter layers. This drastically reduces memory usage and training time while preserving model quality.

```
Full Fine-Tuning  →  ~7B parameters updated
LoRA Fine-Tuning  →  ~millions of parameters updated (adapters only)
```

### 4-bit Quantization (QLoRA)
The base model is loaded in 4-bit precision, compressing it into a fraction of its original VRAM footprint. Unsloth handles this automatically.

### Alpaca Prompt Template
The notebook uses the Alpaca instruction-following format:

```
Below is an instruction that describes a task...

### Instruction:
<task>

### Input:
<context>

### Response:
<answer>
```

---

## 🚀 Quickstart

### Requirements
- Python 3.10+
- CUDA-capable GPU (Google Colab works great)
- ~6 GB VRAM minimum (4-bit mode)

### Installation

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers trl datasets torch
```

### Pipeline at a Glance

```
1. pip install unsloth
        ↓
2. FastLanguageModel.from_pretrained()   ← Load model in 4-bit
        ↓
3. FastLanguageModel.get_peft_model()    ← Attach LoRA adapter
        ↓
4. Prepare dataset + prompt template     ← Format your data
        ↓
5. SFTTrainer.train()                    ← Fine-tune
        ↓
6. FastLanguageModel.for_inference()     ← Switch to inference mode
        ↓
7. model.save_pretrained() / .save_pretrained_gguf()   ← Export
```

---

## 🤖 Model Used

| Setting | Value |
|---------|-------|
| Base Model | `unsloth/Llama-3.2-1B-Instruct` |
| Max Sequence Length | 2048 tokens |
| Quantization | 4-bit (QLoRA) |
| LoRA Rank (`r`) | 16 |
| Dataset | `yahma/alpaca-cleaned` |

### Other Supported Models

| Family | Examples |
|--------|---------|
| Llama | Llama-3.2-1B/3B, Llama-3.1-8B, Llama-3-70B |
| Mistral | Mistral-7B, Mistral-Nemo |
| Phi | Phi-3-mini, Phi-3.5-mini |
| Gemma | Gemma-2-2B/9B/27B |
| Qwen | Qwen-2.5-7B/14B/72B |

---

## 💾 Export Formats

After training, the model can be saved in multiple formats:

| Format | Use Case |
|--------|---------|
| LoRA adapter only | Lightweight, re-merge later |
| Merged 16-bit | Standard HuggingFace deployment |
| GGUF | Local inference with Ollama or llama.cpp |

---

## 📊 Unsloth vs Standard HuggingFace

| Feature | Unsloth | Standard |
|---------|---------|---------|
| Training Speed | ~2x faster | Baseline |
| VRAM Usage | ~60% less | Baseline |
| Min GPU VRAM (7B) | ~6 GB | ~16 GB |
| Accuracy | Same | Same |
| Code changes needed | Minimal | — |

---

## 🔗 Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Docs](https://docs.unsloth.ai)
- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [Alpaca Cleaned Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)
