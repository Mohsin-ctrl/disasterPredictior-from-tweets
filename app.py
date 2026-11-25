import html
import re

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "Mohsan123/disaster-tweet-classifier"
MAX_LEN = 64

st.set_page_config(page_title="Disaster Tweet Detector", page_icon="🚨")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).eval()
    return tokenizer, model


def clean_text(text):
    # exactly the same cleaning as in the notebook (Step 3)
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


st.title("Disaster Tweet Detector")
st.write("Is this tweet about a real disaster? A fine-tuned BERT-base transformer predicts "
         "Disaster or Not Disaster. Built for AIC354, COMSATS University Islamabad, Lahore Campus.")

tweet = st.text_area("Tweet", height=110, placeholder="Type or paste a tweet here...")

if st.button("Predict"):
    if not tweet.strip():
        st.warning("Please type a tweet first.")
    else:
        tokenizer, model = load_model()
        enc = tokenizer(clean_text(tweet), truncation=True, max_length=MAX_LEN,
                        return_tensors="pt", return_token_type_ids=False)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        if int(probs.argmax()) == 1:
            st.error(f"Disaster ({probs[1]:.1%} confidence)")
        else:
            st.success(f"Not Disaster ({probs[0]:.1%} confidence)")
        st.write(f"P(Disaster) = {probs[1]:.1%}")
        st.progress(float(probs[1]))
