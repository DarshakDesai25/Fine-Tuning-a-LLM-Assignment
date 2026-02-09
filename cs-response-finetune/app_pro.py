import time
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Winner from your evaluation on the big dataset: config1
MODEL_CHOICES = {
    "Best Fine-tuned (Bitext) - config1": "outputs/checkpoints/config1/final_model",
    "Fine-tuned (Bitext) - config2": "outputs/checkpoints/config2/final_model",
    "Fine-tuned (Bitext) - config3": "outputs/checkpoints/config3/final_model",
    "Baseline (no fine-tune)": "google/flan-t5-small",
}

# Simple cache so we don't reload weights every request
_loaded = {"name": None, "tok": None, "model": None}

def load_model(choice_name: str):
    path = MODEL_CHOICES[choice_name]
    if _loaded["name"] == choice_name:
        return _loaded["tok"], _loaded["model"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    _loaded.update({"name": choice_name, "tok": tok, "model": model})
    return tok, model

def build_prompt(msg: str, tone: str, ask_details: bool):
    tone_map = {
        "Neutral": "Use a clear, neutral professional tone.",
        "Friendly": "Use a warm, friendly tone without being informal.",
        "Strict Professional": "Use a concise, formal business tone.",
    }

    rule = (
        "Only ask for order/account/email details if the customer message clearly involves order, payment, account, login, or billing."
        if ask_details else
        "Do not ask for email/order/account details. Provide general next steps only."
    )

    # A couple of few-shot examples to reduce generic 'ask for email' replies
    examples = (
        "Examples:\n"
        "Customer message: Who are you?\n"
        "Support response: I’m a customer support assistant. Tell me what issue you need help with and I’ll guide you.\n\n"
        "Customer message: My order is delayed and tracking hasn’t updated.\n"
        "Support response: I’m sorry about the delay. Please share your order ID so I can check the latest status. "
        "In the meantime, try refreshing tracking or checking your order history for updates.\n\n"
    )

    return (
        "You are a customer support agent.\n"
        f"Tone: {tone_map[tone]}\n"
        f"Rule: {rule}\n"
        "Write 1-3 actionable steps. Keep it 2-6 sentences.\n"
        f"{examples}"
        f"Customer message: {msg.strip()}\n"
        "Support response:"
    )

def respond(model_choice, tone, ask_details, temperature, top_p, user_msg, history):
    tok, model = load_model(model_choice)

    msg = (user_msg or "").strip()
    if not msg:
        return "", history or []

    # quick routing for identity questions
    low = msg.lower()
    if "who are you" in low or low in {"who are you", "what are you", "what is this"}:
        reply = "I’m a customer support response assistant. Tell me what issue you’re facing and I’ll help with the next steps."
        history = history or []
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": reply})
        return "", history

    prompt = build_prompt(msg, tone, ask_details)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)

    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=50,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )

    reply = tok.decode(out[0], skip_special_tokens=True).strip()

    history = history or []
    # Gradio 5 expects messages format: list[{"role": "...", "content": "..."}]
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": reply})

    return "", history

def export_chat(history):
    history = history or []
    lines = []
    turn = 0
    i = 0
    while i < len(history):
        role = history[i].get("role")
        if role == "user" and i + 1 < len(history) and history[i+1].get("role") == "assistant":
            turn += 1
            u = history[i].get("content", "")
            a = history[i+1].get("content", "")
            lines.append(f"--- Turn {turn} ---")
            lines.append(f"Customer: {u}")
            lines.append(f"Agent: {a}")
            lines.append("")
            i += 2
        else:
            i += 1

    content = "\n".join(lines).strip() + "\n"
    fname = f"chat_export_{int(time.time())}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    return fname

with gr.Blocks(title="Customer Support Response Assistant") as demo:
    gr.Markdown(
        "# Customer Support Response Assistant\n"
        "Fine-tuned on a large customer-support dataset (Bitext). "
        "Select model, tone, and decoding settings. Export chat for your report/video."
    )

    with gr.Row():
        model_choice = gr.Dropdown(
            list(MODEL_CHOICES.keys()),
            value="Best Fine-tuned (Bitext) - config1",
            label="Model"
        )
        tone = gr.Dropdown(["Neutral", "Friendly", "Strict Professional"], value="Neutral", label="Tone")
        ask_details = gr.Checkbox(value=True, label="Ask for details only if relevant")

    with gr.Row():
        temperature = gr.Slider(0.2, 1.4, value=0.9, step=0.05, label="Temperature")
        top_p = gr.Slider(0.5, 1.0, value=0.92, step=0.02, label="Top-p")

    # Gradio 5 Chatbot uses messages format by default
    chatbot = gr.Chatbot(label="Chat", height=420)

    state = gr.State([])  # will store messages list

    user_msg = gr.Textbox(label="Customer message", placeholder="Type here...", lines=3)

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear chat")
        export_btn = gr.Button("Export chat")

    gr.Examples(
        examples=[
            ["My order is delayed and tracking hasn’t updated."],
            ["I forgot my password and can’t log in."],
            ["I was charged twice for the same subscription."],
            ["How do I cancel my plan?"],
            ["Who are you?"],
        ],
        inputs=[user_msg],
        label="Try examples"
    )

    # Send -> update chatbot -> also sync state
    send.click(
        respond,
        inputs=[model_choice, tone, ask_details, temperature, top_p, user_msg, state],
        outputs=[user_msg, chatbot],
    ).then(lambda chat: chat, inputs=[chatbot], outputs=[state])

    # Clear
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

    # Export
    export_btn.click(export_chat, inputs=[state], outputs=gr.File(label="Download chat export"))

if __name__ == "__main__":
    demo.launch()
