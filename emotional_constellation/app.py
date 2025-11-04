import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageFilter
import hashlib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests

# ----------------------------
# Initialize sentiment analyzer
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_vader()

st.set_page_config(page_title="Emotional Constellation", page_icon="✨", layout="wide")
st.title("🌌 Emotional Constellation")
st.caption("Visualize emotions as a generative night sky — color = emotion type, brightness = intensity. Data → Emotion → Art → Interaction.")

# ----------------------------
# Fetch news data via NewsAPI
# ----------------------------
def fetch_news(api_key, keyword="technology", page_size=30):
    """Fetch latest English news articles containing the keyword."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        articles = data.get("articles", [])
        df = pd.DataFrame([{
            "timestamp": a["publishedAt"][:10],
            "text": (a["title"] or "") + " - " + (a["description"] or ""),
            "source": a["source"]["name"]
        } for a in articles])
        return df
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# ----------------------------
# Helper functions
# ----------------------------
def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return sia.polarity_scores(text)

def classify_emotion(row) -> str:
    """Expanded emotion categories based on sentiment mix."""
    neg, neu, pos, comp = row["neg"], row["neu"], row["pos"], row["compound"]
    # Simple rule-based mapping
    if comp >= 0.6:
        return "joy"
    elif 0.2 <= comp < 0.6:
        return "surprise"
    elif -0.2 < comp < 0.2:
        return "neutral"
    elif -0.6 < comp <= -0.2:
        return "sadness"
    elif comp <= -0.6:
        return "anger"
    # Secondary emotion inference
    if pos > 0.4 and neg > 0.3:
        return "mixed"
    if neg > 0.5 and pos < 0.1:
        return "fear"
    return "neutral"

def seed_from_text(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)

def create_constellation(df: pd.DataFrame, width=1600, height=900, glow=True):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Expanded emotion color map
    color_map = {
        "joy": (1.0, 0.85, 0.2),       # golden yellow
        "surprise": (1.0, 0.5, 0.2),   # orange
        "neutral": (0.8, 0.8, 0.9),    # pale gray
        "sadness": (0.35, 0.5, 1.0),   # soft blue
        "anger": (1.0, 0.25, 0.25),    # red
        "fear": (0.5, 0.2, 0.8),       # purple
        "mixed": (0.3, 1.0, 0.7),      # turquoise
    }

    xs, ys, ss, cs, alphas = [], [], [], [], []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        comp = float(row.get("compound", 0))
        emo = row.get("emotion", "neutral")

        rng = np.random.default_rng(seed_from_text(text))
        x = rng.uniform(0.02, 0.98)
        y = rng.uniform(0.06, 0.94)

        intensity = min(1.0, max(0.0, abs(comp)))
        size = 15 + 220 * intensity**0.8
        alpha = 0.2 + 0.7 * intensity

        xs.append(x); ys.append(y); ss.append(size); alphas.append(alpha)
        cs.append(color_map.get(emo, (0.9, 0.9, 0.9)))

    if xs:
        ax.scatter(xs, ys, s=[s*3.0 for s in ss], c=cs, alpha=[a*0.15 for a in alphas], linewidths=0, marker="o")
        ax.scatter(xs, ys, s=ss, c=cs, alpha=alphas, linewidths=0, marker="o")

    if len(xs) >= 6:
        pts = np.column_stack([xs, ys])
        for i in range(len(pts)):
            d = np.sum((pts - pts[i])**2, axis=1)
            nn = np.argsort(d)[1:3]
            for j in nn:
                ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], linewidth=0.3, alpha=0.15, c="white")

    buf = BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    if glow:
        im = Image.open(buf).convert("RGBA")
        blurred = im.filter(ImageFilter.GaussianBlur(radius=1.2))
        out = Image.alpha_composite(blurred, im)
        out_buf = BytesIO()
        out.save(out_buf, format="PNG")
        out_buf.seek(0)
        return out_buf
    else:
        return buf

# ----------------------------
# UI and interaction
# ----------------------------
with st.expander("ℹ️ Instructions", expanded=False):
    st.markdown("""
    **How to use:**
    1. Choose a data input mode (Upload CSV / Paste text / Fetch news).
    2. Analyze sentiment to map each text into emotion points.
    3. Filter emotions on the sidebar to explore different emotional skies.
    4. Download the current constellation as a PNG.
    """)

st.sidebar.header("Filters")
emotion_options = ["joy", "surprise", "neutral", "sadness", "anger", "fear", "mixed"]
selected_emotions = st.sidebar.multiselect("Select emotions to display:", options=emotion_options, default=emotion_options)

st.sidebar.markdown("---")
st.sidebar.header("Data input")

input_mode = st.sidebar.radio("Select data source", ["Upload CSV", "Paste text", "Fetch news"], index=0)
df = pd.DataFrame()

if input_mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload a CSV file (must contain a 'text' column)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.sidebar.error("Failed to read CSV. Check encoding and delimiter.")

elif input_mode == "Paste text":
    user_text = st.sidebar.text_area("Paste multiple lines (one text per line):", height=200)
    if st.sidebar.button("Add to dataset", use_container_width=True):
        rows = [t for t in user_text.splitlines() if t.strip()]
        df = pd.DataFrame({"text": rows})
        df["timestamp"] = pd.Timestamp.today().date().astype(str)

elif input_mode == "Fetch news":
    keyword = st.sidebar.text_input("Enter keyword (e.g., economy / technology / happiness)", "technology")
    if st.sidebar.button("Fetch from NewsAPI", use_container_width=True):
        api_key = st.secrets.get("NEWS_API_KEY", "")
        if not api_key:
            st.sidebar.error("⚠️ Missing API key. Please add NEWS_API_KEY in Streamlit Secrets.")
        else:
            df = fetch_news(api_key, keyword=keyword)

if df.empty:
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("Using sample_data.csv since no data was provided.")
    except Exception:
        st.error("No data found and sample_data.csv missing.")
        st.stop()

if "text" not in df.columns:
    st.error("The dataset must include a 'text' column.")
    st.stop()

with st.spinner("Analyzing sentiment..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion, axis=1)

df = df[df["emotion"].isin(selected_emotions)].reset_index(drop=True)

left, right = st.columns([0.58, 0.42])
with left:
    st.subheader("⭐ Emotional Constellation")
    if df.empty:
        st.warning("No data points under current filters.")
    else:
        img_buf = create_constellation(df, width=1600, height=900, glow=True)
        st.image(img_buf, caption="Emotional Constellation", use_column_width=True)
        st.download_button("💾 Download current sky as PNG", data=img_buf, file_name="emotional_constellation.png", mime="image/png")

with right:
    st.subheader("📊 Data & Sentiment Details")
    st.dataframe(
        df[["text", "compound", "pos", "neu", "neg", "emotion"] + ([c for c in ["timestamp","source"] if c in df.columns])],
        use_container_width=True, height=420
    )

st.markdown("---")
st.caption("Made with ❤️  — Data → Emotion → Generative Art → Streamlit © 2025")
