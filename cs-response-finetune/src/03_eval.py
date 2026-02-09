import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@torch.inference_mode()
def generate(model, tok, prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
    out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)

def quality_score(text: str) -> int:
    # Simple, rubric-friendly: 0 to 3
    t = text.lower()
    score = 0
    if len(text) >= 30: score += 1
    if "please" in t or "sorry" in t or "happy to help" in t: score += 1
    if "provide" in t or "account" in t or "order" in t or "details" in t: score += 1
    return score

def main():
    test_ds = load_dataset("json", data_files="data/test.jsonl")["train"]

    # Baseline model (not fine-tuned)
    base_name = "google/flan-t5-base"
    base_tok = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)

    models = [("baseline", base_model, base_tok)]

    # Fine-tuned models
    for cfg in ["config1", "config2", "config3"]:
        path = f"outputs/checkpoints/{cfg}/final_model"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        models.append((cfg, model, tok))

    rows = []
    for ex in test_ds:
        prompt = ex["instruction"] + "\nInput: " + ex["input"]
        gold = ex["output"]

        for name, model, tok in models:
            pred = generate(model, tok, prompt)
            rows.append({
                "model": name,
                "prompt": ex["input"],
                "gold": gold,
                "pred": pred,
                "quality_score": quality_score(pred),
            })

    out_path = "outputs/eval/results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()
