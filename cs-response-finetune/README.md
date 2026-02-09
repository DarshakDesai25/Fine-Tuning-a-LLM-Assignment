cat > README.md << 'EOF'
# Customer Support Response Assistant (LLM Fine-Tuning)

This project fine-tunes a pre-trained sequence-to-sequence LLM to generate professional customer-support responses. It includes:
- Dataset preparation (cleaning + train/val/test split)
- Fine-tuning with 3 hyperparameter configurations
- Baseline vs fine-tuned evaluation
- Error analysis
- A Gradio UI demo with tone + decoding controls + chat export

## Dataset
Primary dataset: Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face).
We convert the dataset into instruction-style JSONL for fine-tuning.

## Model
Base model: `google/flan-t5-small` (chosen for CPU-friendly training and fast iteration).

## Setup

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

