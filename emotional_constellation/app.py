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
import requests  # ← 新增，用于 NewsAPI

# -----------------------
# 初始化情感分析器
# -----------------------
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

# -----------------------
# 📰 从 NewsAPI 抓取新闻
# -----------------------
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
            st.warning("NewsAPI 返回错误: " + str(data.get("message")))
            return pd.DataFrame()
        articles = data.get("articles", [])
        df = pd.DataFrame([{
            "timestamp": a["publishedAt"][:10],
            "text": a["title"] + " - " + (a["description"] or ""),
            "source": a["source"]["name"]
        } for a in articles])
        return df
    except Exception as e:
        st.error(f"请求 NewsAPI 时出错: {e}")
        return pd.DataFrame()

# -----------------------
# 函数定义
# -----------------------
def analyze_sentiment(text: str) -> dict:
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
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)

def create_constellation(df: pd.DataFrame, width=1600, height=900, glow=True):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    color_map = {
        "positive": (1.0, 0.93, 0.2),
        "neutral": (0.8, 0.8, 0.9),
        "negative": (0.3, 0.55, 1.0),
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
        size = 20 + 180 * intensity**0.8
        alpha = 0.25 + 0.65 * intensity

        xs.append(x); ys.append(y); ss.append(size); alphas.append(alpha)
        cs.append(color_map.get(emo, (0.9, 0.9, 0.9)))

    if xs:
        ax.scatter(xs, ys, s=[s*3.0 for s in ss], c=cs, alpha=[a*0.18 for a in alphas], linewidths=0, marker="o")
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

# -----------------------
# 界面与交互
# -----------------------
with st.expander("ℹ️ 使用说明 / How to use", expanded=False):
    st.markdown("""
    1. 选择 **数据输入方式**（上传 CSV / 粘贴文本 / 抓取新闻）。  
    2. 点击 **分析情绪**，得到每条文本的极性分值。  
    3. 使用左侧过滤器选择情绪类型或时间范围，查看生成的“情绪星空”。  
    4. 点击 **导出图像** 保存当前星空为 PNG。
    """)

st.sidebar.header("过滤 / Filters")
emotion_options = ["positive", "neutral", "negative"]
selected_emotions = st.sidebar.multiselect("情绪类型 / Emotion types", options=emotion_options, default=emotion_options)
st.sidebar.markdown("---")
st.sidebar.header("数据输入 / Data input")

input_mode = st.sidebar.radio("选择数据来源", ["上传 CSV", "粘贴文本", "抓取新闻"], index=0)
df = pd.DataFrame()

if input_mode == "上传 CSV":
    up = st.sidebar.file_uploader("上传 CSV (包含 text 列)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.sidebar.error("无法读取该 CSV，请确认编码与分隔符。")

elif input_mode == "粘贴文本":
    user_text = st.sidebar.text_area("粘贴多行文本（每行一条记录）", height=200)
    if st.sidebar.button("添加到数据集", use_container_width=True):
        rows = [t for t in user_text.splitlines() if t.strip()]
        df = pd.DataFrame({"text": rows})
        df["timestamp"] = pd.Timestamp.today().date().astype(str)

elif input_mode == "抓取新闻":
    keyword = st.sidebar.text_input("输入关键词（英文，如 technology / economy / happiness）", "technology")
    if st.sidebar.button("从 NewsAPI 抓取新闻", use_container_width=True):
        api_key = st.secrets.get("NEWS_API_KEY", "")
        if not api_key:
            st.sidebar.error("⚠️ 未检测到 API Key，请在 Streamlit Secrets 中添加 NEWS_API_KEY")
        else:
            df = fetch_news(api_key, keyword=keyword)

if df.empty:
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("未提供数据，使用示例数据 sample_data.csv。")
    except Exception:
        st.error("未能加载示例数据。")
        st.stop()

if "text" not in df.columns:
    st.error("数据中必须包含 `text` 列。")
    st.stop()

with st.spinner("分析情绪中..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df["compound"].apply(classify_emotion)

df = df[df["emotion"].isin(selected_emotions)].reset_index(drop=True)

left, right = st.columns([0.58, 0.42])
with left:
    st.subheader("⭐ 情绪星空 / Constellation")
    if df.empty:
        st.warning("当前过滤条件下没有数据点。")
    else:
        img_buf = create_constellation(df, width=1600, height=900, glow=True)
        st.image(img_buf, caption="Emotional Constellation", use_column_width=True)
        st.download_button("💾 导出当前星空为 PNG", data=img_buf, file_name="emotional_constellation.png", mime="image/png")

with right:
    st.subheader("📊 数据与情绪 / Data & Sentiment")
    st.dataframe(df[["text", "compound", "pos", "neu", "neg", "emotion"] + ([c for c in ["timestamp","source"] if c in df.columns])],
                 use_container_width=True, height=420)

st.markdown("---")
st.caption("Made with ❤️  Data → Sentiment → Generative Art → Streamlit.  © 2025")
