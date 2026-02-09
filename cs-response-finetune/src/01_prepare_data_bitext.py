import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT_DIR = Path("data_bitext")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s: str) -> str:
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(ds["train"])

    # Bitext columns include 'instruction' (intent), 'category', 'intent', 'response', etc. :contentReference[oaicite:2]{index=2}
    print("Columns:", df.columns.tolist())
    print(df.head(2))

    # Commonly present:
    # - 'instruction' or similar: customer utterance
    # - 'response': target answer
    if "instruction" in df.columns:
        prompt_col = "instruction"
    elif "utterance" in df.columns:
        prompt_col = "utterance"
    else:
        # fallback: first text-ish col
        prompt_col = df.columns[0]

    if "response" not in df.columns:
        raise ValueError("Expected a 'response' column in Bitext dataset.")

    df = df[[prompt_col, "response"]].rename(columns={prompt_col: "prompt", "response": "response"})
    df["prompt"] = df["prompt"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)

    df = df.dropna().drop_duplicates()
    df = df[(df["prompt"].str.len() >= 5) & (df["response"].str.len() >= 10)].reset_index(drop=True)

    # Split: train/val/test (90/5/5) — with large data you can do smaller val/test
    n = len(df)
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)

    train_df = df.iloc[:train_end].sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df.iloc[val_end:].sample(frac=1, random_state=42).reset_index(drop=True)

    def to_jsonl(d: pd.DataFrame, path: Path):
        with path.open("w", encoding="utf-8") as f:
            for _, r in d.iterrows():
                obj = {
                    "instruction": (
                        "You are a helpful professional customer support agent. "
                        "Answer the customer directly. "
                        "Only ask for order/account/email details if relevant to the issue."
                    ),
                    "input": r["prompt"],
                    "output": r["response"],
                }
                f.write(pd.Series(obj).to_json() + "\n")

    to_jsonl(train_df, OUT_DIR / "train.jsonl")
    to_jsonl(val_df, OUT_DIR / "val.jsonl")
    to_jsonl(test_df, OUT_DIR / "test.jsonl")

    print("Saved:", OUT_DIR)
    print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

if __name__ == "__main__":
    main()
