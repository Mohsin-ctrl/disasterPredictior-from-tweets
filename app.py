import html
import re

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "Mohsan123/disaster-tweet-classifier"
MAX_LEN = 64
LABELS = {0: "Not Disaster", 1: "Disaster"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).eval()


def clean_text(text):
    # exactly the same cleaning as in the notebook (Step 3)
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(tweet):
    if not tweet or not tweet.strip():
        return {}
    enc = tokenizer(clean_text(tweet), truncation=True, max_length=MAX_LEN,
                    return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(2)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Tweet", placeholder="Type or paste a tweet here..."),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Disaster Tweet Detector",
    description="Is this tweet about a real disaster? A fine-tuned BERT-base transformer predicts "
                "Disaster or Not Disaster. Built for AIC354 (Machine Learning Fundamentals), COMSATS University Islamabad, Lahore Campus.",
    examples=[
        ["Forest fire near the national park, residents told to leave immediately"],
        ["A magnitude 6.1 earthquake just hit the coast, buildings are damaged"],
        ["This new album is fire, I can't stop listening to it"],
        ["Had a blast at the beach with my family this weekend"],
    ],
)

if __name__ == "__main__":
    demo.launch()
