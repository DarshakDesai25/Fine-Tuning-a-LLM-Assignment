import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "outputs/checkpoints/config3/final_model"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def respond(customer_message):
    customer_message = (customer_message or "").strip()

    prompt = (
        "You are a helpful professional customer support agent.\n"
        "Rules:\n"
        "1) Answer the customer's question directly.\n"
        "2) Only ask for email/order/account details if the customer mentions an order, payment, account, login, or billing issue.\n"
        "3) Keep it concise (2-6 sentences).\n"
        f"Customer message: {customer_message}\n"
        "Support response:"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)

    out = model.generate(
        **inputs,
        max_new_tokens=140,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    return tok.decode(out[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Customer message", lines=4),
    outputs=gr.Textbox(label="Model response", lines=6),
    title="Customer Support Response Assistant (Fine-tuned)"
)

if __name__ == "__main__":
    demo.launch()
