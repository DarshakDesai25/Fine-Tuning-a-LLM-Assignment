import pandas as pd

def main():
    df = pd.read_csv("outputs/eval/results.csv")

    worst = df.sort_values("quality_score").head(12)

    print("\n=== 12 WORST EXAMPLES (lowest quality_score) ===\n")
    for _, r in worst.iterrows():
        print("MODEL:", r["model"])
        print("SCORE:", r["quality_score"])
        print("PROMPT:", r["prompt"])
        print("GOLD:", r["gold"])
        print("PRED:", r["pred"])
        print("-" * 90)

    print("\nPatterns to write in your report:")
    print("1) Generic response not tailored to prompt")
    print("2) Missing request for required details")
    print("3) Hallucination of account/order info")
    print("4) Overly short or incomplete answer")
    print("\nSuggested improvements:")
    print("- Add more diverse examples")
    print("- Use stricter instructions")
    print("- Adjust decoding parameters")
    print("- Try larger base model or LoRA")

if __name__ == "__main__":
    main()
