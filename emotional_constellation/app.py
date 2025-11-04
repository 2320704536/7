
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

# Ensure VADER is available on first run (cached by Streamlit)
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_vader()

st.set_page_config(page_title="情绪星空 Emotional Constellation", page_icon="✨", layout="wide")

st.title("🌌 情绪星空 (Emotional Constellation)")
st.caption("将文本情绪映射为动态星空：颜色=情绪类别，亮度/大小=情绪强度。Data → Art → Interaction.")

with st.expander("ℹ️ 使用说明 / How to use", expanded=False):
    st.markdown("""
    1. 选择 **数据输入方式**（上传 CSV 或粘贴文本）。  
    2. 点击 **分析情绪**，得到每条文本的极性分值。  
    3. 使用左侧过滤器选择时间范围、情绪类型，查看生成的“情绪星空”。  
    4. 点击 **导出图像** 保存当前星空为 PNG。
    """)

# Sidebar controls
st.sidebar.header("过滤 / Filters")
emotion_options = ["positive", "neutral", "negative"]
selected_emotions = st.sidebar.multiselect("情绪类型 / Emotion types", options=emotion_options, default=emotion_options)

st.sidebar.markdown("---")
st.sidebar.header("数据输入 / Data input")

input_mode = st.sidebar.radio("选择数据来源", ["上传 CSV", "粘贴文本"], index=0)

def analyze_sentiment(text: str) -> dict:
    # Basic fallback for empty string
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    scores = sia.polarity_scores(text)
    return scores

def classify_emotion(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def seed_from_text(text: str) -> int:
    # Deterministic seed from text hash for stable layout per row
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)

def create_constellation(df: pd.DataFrame, width=1600, height=900, glow=True):
    # Prepare canvas via matplotlib
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Base star field parameters
    xs, ys, ss, cs, alphas = [], [], [], [], []

    # Map emotions to colors
    color_map = {
        "positive": (1.0, 0.93, 0.2),  # yellow-ish
        "neutral": (0.8, 0.8, 0.9),    # pale gray
        "negative": (0.3, 0.55, 1.0),  # bluish
    }

    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        comp = float(row.get("compound", 0))
        emo = row.get("emotion", "neutral")

        # Deterministic pos from text hash
        rng = np.random.default_rng(seed_from_text(text))
        x = rng.uniform(0.02, 0.98)
        y = rng.uniform(0.06, 0.94)

        # Size and alpha from intensity
        intensity = min(1.0, max(0.0, abs(comp)))
        size = 20 + 180 * intensity**0.8
        alpha = 0.25 + 0.65 * intensity

        xs.append(x); ys.append(y); ss.append(size); alphas.append(alpha)
        cs.append(color_map.get(emo, (0.9, 0.9, 0.9)))

    # Two-pass draw for subtle "glow": large faint + normal
    if len(xs) > 0:
        ax.scatter(xs, ys, s=[s*3.0 for s in ss], c=cs, alpha=[a*0.18 for a in alphas], linewidths=0, marker="o")
        ax.scatter(xs, ys, s=ss, c=cs, alpha=alphas, linewidths=0, marker="o")

    # Constellation connecting nearest neighbors (optional aesthetic)
    if len(xs) >= 6:
        pts = np.column_stack([xs, ys])
        # Simple heuristic: connect each point to its 2 nearest neighbors
        for i in range(len(pts)):
            d = np.sum((pts - pts[i])**2, axis=1)
            nn = np.argsort(d)[1:3]
            for j in nn:
                ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]],
                        linewidth=0.3, alpha=0.15, c="white")

    buf = BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    if glow:
        # Apply slight Gaussian blur for overall glow
        im = Image.open(buf).convert("RGBA")
        blurred = im.filter(ImageFilter.GaussianBlur(radius=1.2))
        out = Image.alpha_composite(blurred, im)
        out_buf = BytesIO()
        out.save(out_buf, format="PNG")
        out_buf.seek(0)
        return out_buf
    else:
        return buf

# Data ingestion
df = pd.DataFrame()

if input_mode == "上传 CSV":
    up = st.sidebar.file_uploader("上传 CSV (需要包含 text 列，可选 timestamp / source)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.sidebar.error("无法读取该 CSV，请确认编码与分隔符。")
else:
    user_text = st.sidebar.text_area("粘贴多行文本（每行一条记录）", height=200, placeholder="例：\nI love the sunset.\n今天有点累，但我会坚持。\nThe new feature is awesome!")
    if st.sidebar.button("添加到数据集", use_container_width=True):
        rows = [t for t in user_text.splitlines() if t.strip()]
        df = pd.DataFrame({"text": rows})
        # Add timestamp as today
        df["timestamp"] = pd.Timestamp.today().date().astype(str)

if df.empty:
    # Provide sample
    with st.container(border=True):
        st.subheader("示例数据 / Sample data")
        st.write("未上传数据时，自动载入 `sample_data.csv`。")
    try:
        df = pd.read_csv("sample_data.csv")
    except Exception:
        # Fallback if file not found
        df = pd.DataFrame({
            "timestamp": ["2025-10-28","2025-10-29","2025-10-30"],
            "text": ["I can't believe how beautiful the sky is tonight!",
                     "今天的作业太多了，我有点崩溃。",
                     "We did it! Our team finally shipped the feature!"],
            "source": ["news","tweet","news"],
        })

# Parse timestamp if exists
if "timestamp" in df.columns:
    def try_parse(x):
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return pd.NaT
    ts = df["timestamp"].apply(try_parse)
    if ts.notna().any():
        df["ts_date"] = ts
        min_d = ts.min()
        max_d = ts.max()
        with st.sidebar:
            if pd.isna(min_d) or pd.isna(max_d):
                st.info("检测到部分日期不可解析，将忽略时间过滤。")
                date_filter = None
            else:
                date_filter = st.date_input("时间范围", value=(min_d, max_d))
        if date_filter:
            start_d, end_d = date_filter
            mask = (df["ts_date"] >= start_d) & (df["ts_date"] <= end_d)
            df = df.loc[mask].copy()

# Sentiment analysis
if "text" not in df.columns:
    st.error("数据中必须包含 `text` 列。")
    st.stop()

with st.spinner("分析情绪中..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df["compound"].apply(classify_emotion)

# Filter by emotion
df = df[df["emotion"].isin(selected_emotions)].reset_index(drop=True)

# Layout
left, right = st.columns([0.58, 0.42])

with left:
    st.subheader("⭐ 情绪星空 / Constellation")
    if df.empty:
        st.warning("当前过滤条件下没有数据点，请调整情绪/时间范围。")
    else:
        img_buf = create_constellation(df, width=1600, height=900, glow=True)
        st.image(img_buf, caption="Emotional Constellation", use_column_width=True)

        st.download_button("💾 导出当前星空为 PNG", data=img_buf, file_name="emotional_constellation.png", mime="image/png")

with right:
    st.subheader("📊 数据与情绪 / Data & Sentiment")
    st.dataframe(
        df[["text", "compound", "pos", "neu", "neg", "emotion"] + ([c for c in ["timestamp","source"] if c in df.columns])],
        use_container_width=True, height=420
    )

st.markdown("---")
st.caption("Made with ❤️  Data → Sentiment → Generative Art → Streamlit.  © 2025")
