import html
import re

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ------------------------------------------------------------------ settings (same as the notebook)
MODEL_REPO = "Mohsan123/disaster-tweet-classifier"
MAX_LEN = 64
MAX_BATCH = 50
LABELS = {0: "Not Disaster", 1: "Disaster"}

EXAMPLES = [
    (" Forest fire", "Forest fire near the national park, residents told to leave immediately"),
    (" Earthquake", "A magnitude 6.1 earthquake just hit the coast, buildings are damaged"),
    (" Flood", "Flash flood warning: roads are underwater and rescue teams are on the way"),
    (" Album is fire", "This new album is fire, I can't stop listening to it"),
    (" Beach day", "Had a blast at the beach with my family this weekend"),
    (" Phone exploded", "My phone just exploded with notifications lol"),
]

st.set_page_config(page_title="Disaster Tweet Detector", page_icon="", layout="centered")


# ------------------------------------------------------------------ model and prediction
@st.cache_resource(show_spinner="Loading the model (the first start can take a minute)...")
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


def predict(texts):
    # returns an array with one row per tweet: [P(Not Disaster), P(Disaster)]
    tokenizer, model = load_model()
    enc = tokenizer([clean_text(t) for t in texts], truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    return probs.numpy()


def is_truncated(text):
    tokenizer, _ = load_model()
    return len(tokenizer(clean_text(text))["input_ids"]) > MAX_LEN


# ------------------------------------------------------------------ session state and callbacks
st.session_state.setdefault("tweet_input", "")
st.session_state.setdefault("history", [])


def use_example(text):
    st.session_state["tweet_input"] = text
    st.session_state["run_single"] = True


def clear_history():
    st.session_state["history"] = []


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About this app")
    st.write("A fine-tuned **BERT-base** transformer reads a tweet and predicts whether it is "
             "about a **real disaster**.")

    st.subheader("Model performance")
    st.caption("Measured on 692 tweets that the model never saw during training.")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "82.7%")
    c2.metric("F1 (Disaster)", "0.779")
    c1.metric("Precision", "0.809")
    c2.metric("Recall", "0.752")

    st.subheader("How it works")
    st.markdown(
        "1. The tweet is cleaned (links and @names are replaced).\n"
        "2. The BERT tokenizer turns it into tokens.\n"
        "3. The fine-tuned model gives a probability for each class."
    )

    st.subheader("Good to know")
    st.markdown(
        "- Trained on about 5,500 labelled tweets (Kaggle *NLP with Disaster Tweets*).\n"
        "- Jokes and metaphors (*\"this album is fire\"*) can still fool it.\n"
        "- It is a student project, **not** an official alert system."
    )
    st.caption("AIC354 – Machine Learning Fundamentals · COMSATS University Islamabad, Lahore Campus")


# ------------------------------------------------------------------ main page
st.title("🚨 Disaster Tweet Detector")
st.write("Is this tweet about a **real disaster**? Type a tweet, pick an example, or check many tweets at once.")

tab_single, tab_batch = st.tabs(["Single tweet", "Many tweets"])

# ---- Tab 1: one tweet
with tab_single:
    st.write("**Try an example:**")
    cols = st.columns(3)
    for k, (label, text) in enumerate(EXAMPLES):
        cols[k % 3].button(label, key=f"example_{k}", on_click=use_example, args=(text,))

    st.text_area("Tweet", key="tweet_input", height=110, placeholder="Type or paste a tweet here...")
    clicked = st.button("Predict", type="primary")
    auto_run = st.session_state.pop("run_single", False)

    if clicked or auto_run:
        tweet = st.session_state["tweet_input"]
        if not tweet.strip():
            st.warning("Please type a tweet first.")
        else:
            probs = predict([tweet])[0]
            p_disaster = float(probs[1])
            label = LABELS[int(probs.argmax())]
            confidence = float(probs.max())

            if label == "Disaster":
                st.error(f"🚨 **Disaster** – {confidence:.1%} confidence")
            else:
                st.success(f"✅ **Not Disaster** – {confidence:.1%} confidence")

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", label)
            m2.metric("Confidence", f"{confidence:.1%}")
            m3.metric("P(Disaster)", f"{p_disaster:.1%}")

            st.bar_chart(pd.DataFrame({"Probability": [float(probs[0]), p_disaster]},
                                      index=["Not Disaster", "Disaster"]))

            if confidence < 0.65:
                st.info("The model is not very sure about this tweet. Treat the prediction with care.")
            if is_truncated(tweet):
                st.caption(f"Note: this tweet is longer than {MAX_LEN} tokens, so only the first part was used.")

            st.session_state["history"] = ([{"Tweet": tweet[:90], "Prediction": label,
                                             "P(Disaster)": f"{p_disaster:.1%}"}]
                                           + st.session_state["history"])[:10]

    if st.session_state["history"]:
        with st.expander("Recent predictions"):
            st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)
            st.button("Clear history", on_click=clear_history)

# ---- Tab 2: many tweets
with tab_batch:
    st.write(f"Paste up to **{MAX_BATCH} tweets**, one tweet per line.")
    batch_text = st.text_area("Tweets (one per line)", key="batch_input", height=180,
                              placeholder="Forest fire near the national park...\nHad a blast at the beach...")

    if st.button("Predict all tweets", type="primary"):
        tweets = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not tweets:
            st.warning("Please paste at least one tweet.")
        else:
            if len(tweets) > MAX_BATCH:
                st.warning(f"Only the first {MAX_BATCH} tweets are used.")
                tweets = tweets[:MAX_BATCH]
            probs = predict(tweets)
            result = pd.DataFrame({
                "Tweet": tweets,
                "Prediction": [LABELS[int(p.argmax())] for p in probs],
                "P(Disaster)": [round(float(p[1]), 3) for p in probs],
            })
            n_disaster = int((result["Prediction"] == "Disaster").sum())
            st.write(f"**{n_disaster}** of {len(result)} tweets are predicted as *Disaster*.")
            st.dataframe(result, hide_index=True)
            st.download_button("Download results (CSV)", result.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")

st.divider()
st.caption("Predictions come from a machine learning model trained on noisy, human-labelled tweets. "
           "Do not use them for emergency decisions.")
