import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(s: str) -> str:
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    return s


def augment_prompt(p: str) -> list[str]:
    # simple prompt variants to increase data size without using another model
    templates = [
        "{p}",
        "Customer asked: {p}",
        "Question: {p}",
        "Support request: {p}",
        "Help needed: {p}",
    ]
    return [t.format(p=p) for t in templates]


def main():
    # Load dataset from Hugging Face
    ds = load_dataset("Kaludi/Customer-Support-Responses")
    df = pd.DataFrame(ds["train"])

    # Print columns so we can confirm schema
    print("Columns found:", df.columns.tolist())
    print("Sample row:\n", df.head(1))

    # Use first two columns as (prompt, response)
    if len(df.columns) < 2:
        raise ValueError("Expected at least 2 columns for prompt/response.")

    col_q, col_a = df.columns[0], df.columns[1]
    df = df[[col_q, col_a]].rename(
        columns={col_q: "prompt", col_a: "response"})

    # Clean
    df["prompt"] = df["prompt"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)

    # Drop duplicates / junk
    df = df.drop_duplicates()
    df = df[(df["prompt"].str.len() >= 10) & (
        df["response"].str.len() >= 20)].reset_index(drop=True)

    # Augment prompts (important because dataset is small)
    aug_rows = []
    for _, r in df.iterrows():
        for ap in augment_prompt(r["prompt"]):
            aug_rows.append({"prompt": ap, "response": r["response"]})
    aug_df = pd.DataFrame(aug_rows).drop_duplicates()

    full_df = aug_df.reset_index(drop=True)

    # Split 80/10/10
    train_df, temp_df = train_test_split(
        full_df, test_size=0.2, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, shuffle=True)

    def to_jsonl(d: pd.DataFrame, path: Path):
        with path.open("w", encoding="utf-8") as f:
            for _, r in d.iterrows():
                obj = {
                    "instruction": "Write a helpful, professional customer support response.",
                    "input": r["prompt"],
                    "output": r["response"],
                }
                f.write(pd.Series(obj).to_json() + "\n")

    to_jsonl(train_df, OUT_DIR / "train.jsonl")
    to_jsonl(val_df, OUT_DIR / "val.jsonl")
    to_jsonl(test_df, OUT_DIR / "test.jsonl")

    print("\nSaved files:")
    print(" - data/train.jsonl", len(train_df))
    print(" - data/val.jsonl  ", len(val_df))
    print(" - data/test.jsonl ", len(test_df))


if __name__ == "__main__":
    main()
