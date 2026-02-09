import argparse
from pathlib import Path
import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())

    model_name = cfg["model_name"]
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    batch_size = int(cfg["batch_size"])
    max_in = int(cfg["max_input_len"])
    max_out = int(cfg["max_target_len"])

    # NEW: allow different dataset folders (data or data_bitext)
    data_dir = cfg.get("data_dir", "data")

    train_ds = load_dataset("json", data_files=f"{data_dir}/train.jsonl")["train"]
    val_ds = load_dataset("json", data_files=f"{data_dir}/val.jsonl")["train"]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize(ex):
        src = ex["instruction"] + "\nInput: " + ex["input"]
        x = tok(src, truncation=True, padding="max_length", max_length=max_in)
        y = tok(ex["output"], truncation=True, padding="max_length", max_length=max_out)
        x["labels"] = y["input_ids"]
        return x

    train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize, remove_columns=val_ds.column_names)

    out_dir = Path("outputs/checkpoints") / Path(config_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",      # IMPORTANT for your transformers version
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print("✅ Saved final model to:", final_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
