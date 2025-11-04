import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageFilter
import hashlib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# -------------------------------------------------------
# App setup
# -------------------------------------------------------
st.set_page_config(page_title="Emotional Constellation — Full & Balanced", page_icon="✨", layout="wide")
st.title("🌌 Emotional Constellation — Full & Balanced Edition")
st.caption("English UI • Auto color enrichment • Auto density balance • Layer controls • Always colorful & full visuals.")

# -------------------------------------------------------
# Resources
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_vader()

# -------------------------------------------------------
# NewsAPI
# -------------------------------------------------------
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        articles = data.get("articles", [])
        rows = []
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            txt = (title + " - " + desc).strip(" -")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt,
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# -------------------------------------------------------
# Palettes (vivid & pastel), >20 emotions
# -------------------------------------------------------
PALETTES = {
    "vivid": {
        "joy":        (1.00, 0.84, 0.20),
        "love":       (1.00, 0.45, 0.60),
        "pride":      (0.95, 0.30, 0.75),
        "hope":       (0.40, 0.90, 0.40),
        "curiosity":  (0.25, 0.80, 1.00),
        "calm":       (0.55, 0.80, 1.00),
        "surprise":   (1.00, 0.55, 0.20),
        "neutral":    (0.82, 0.84, 0.92),
        "sadness":    (0.35, 0.55, 1.00),
        "anger":      (1.00, 0.25, 0.25),
        "fear":       (0.55, 0.25, 0.85),
        "disgust":    (0.55, 0.80, 0.25),
        "anxiety":    (0.95, 0.75, 0.20),
        "boredom":    (0.70, 0.70, 0.75),
        "nostalgia":  (1.00, 0.75, 0.45),
        "gratitude":  (0.35, 1.00, 0.75),
        "awe":        (0.60, 0.60, 1.00),
        "trust":      (0.25, 0.95, 0.85),
        "confusion":  (0.90, 0.65, 1.00),
        "mixed":      (0.30, 1.00, 0.70),
    },
    "pastel": {
        "joy":        (0.99, 0.92, 0.60),
        "love":       (1.00, 0.70, 0.78),
        "pride":      (0.98, 0.70, 0.90),
        "hope":       (0.75, 0.95, 0.75),
        "curiosity":  (0.70, 0.90, 1.00),
        "calm":       (0.78, 0.90, 1.00),
        "surprise":   (1.00, 0.78, 0.60),
        "neutral":    (0.90, 0.92, 0.96),
        "sadness":    (0.70, 0.80, 1.00),
        "anger":      (1.00, 0.60, 0.60),
        "fear":       (0.80, 0.70, 0.95),
        "disgust":    (0.80, 0.92, 0.70),
        "anxiety":    (0.98, 0.88, 0.65),
        "boredom":    (0.82, 0.82, 0.86),
        "nostalgia":  (1.00, 0.88, 0.70),
        "gratitude":  (0.70, 1.00, 0.88),
        "awe":        (0.82, 0.82, 1.00),
        "trust":      (0.70, 0.98, 0.92),
        "confusion":  (0.95, 0.85, 1.00),
        "mixed":      (0.75, 1.00, 0.88),
    }
}
ALL_EMOTIONS = list(PALETTES["vivid"].keys())

# -------------------------------------------------------
# Sentiment + expanded emotion mapping
# -------------------------------------------------------
def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row) -> str:
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    # Primary cues
    if comp >= 0.7 and pos > 0.5:             return "joy"
    if comp >= 0.55 and pos > 0.45:           return "love"
    if comp >= 0.45 and pos > 0.40:           return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30:    return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5:    return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5:     return "surprise"

    if comp <= -0.65 and neg > 0.5:           return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45:  return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"

    # Secondary cues
    if neg > 0.5 and neu > 0.3:               return "anxiety"
    if neg > 0.45 and pos < 0.1:              return "disgust"
    if neu > 0.75 and abs(comp) < 0.1:        return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25:             return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15:    return "awe"

    return "neutral"

def seed_from_text(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)

# -------------------------------------------------------
# Visual utilities
# -------------------------------------------------------
def to_uint8_rgb(rgb):
    return tuple(int(255 * max(0, min(1, c))) for c in rgb)

def draw_background_gradient(width, height, top_rgb=(0.03, 0.05, 0.12), bottom_rgb=(0.00, 0.00, 0.00)):
    top = np.array(to_uint8_rgb(top_rgb), dtype=np.uint8)
    bottom = np.array(to_uint8_rgb(bottom_rgb), dtype=np.uint8)
    alpha = np.linspace(0, 1, height).reshape(height, 1)
    grad = (top * (1 - alpha) + bottom * alpha).astype(np.uint8)
    img = np.repeat(grad, repeats=width, axis=1).reshape(height, width, 3)
    return Image.fromarray(img, mode="RGB")

def stratified_positions(n, rng):
    """Blue-noise-ish stratified sampling for nicer layouts than pure uniform."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    xs, ys = [], []
    for r in range(rows):
        for c in range(cols):
            if len(xs) >= n:
                break
            x0 = (c + rng.uniform(0.15, 0.85)) / cols
            y0 = (r + rng.uniform(0.15, 0.85)) / rows
            xs.append(np.clip(x0, 0.02, 0.98))
            ys.append(np.clip(y0, 0.06, 0.94))
    return np.array(xs), np.array(ys)

# -------------------------------------------------------
# Color enrichment + density balancing (visual layer only)
# -------------------------------------------------------
def enrich_and_balance(df, palette_name, rng, min_emotions=8, target_points=800):
    """
    Returns a new DataFrame 'df_viz' for rendering:
    - Ensures at least 'min_emotions' distinct colors by injecting visual-only categories.
    - Upsamples to 'target_points' rows by duplicating with jitter metadata (visual-only).
    Original df is not modified; sentiment/emotion columns are kept for reference, but we use 'viz_emotion'.
    """
    df = df.copy()
    cmap = PALETTES.get(palette_name, PALETTES["vivid"])

    # Use computed emotion as base
    if "emotion" not in df.columns:
        raise ValueError("missing 'emotion' column before enrichment")

    present = df["emotion"].unique().tolist()
    need = max(0, min_emotions - len(present))
    if need > 0:
        # choose additional hues not present
        extra = [e for e in cmap.keys() if e not in present]
        rng.shuffle(extra)
        chosen = extra[:need] if need <= len(extra) else extra
        # inject a small number of visual-only rows per extra category
        inject_n = max(6, len(df) // max(6, len(present)))  # scale with data size
        fake_rows = []
        for emo in chosen:
            for _ in range(inject_n):
                idx = rng.integers(0, len(df))
                row = df.iloc[idx].to_dict()
                row["viz_emotion"] = emo
                row["_is_visual_only"] = True
                fake_rows.append(row)
        df_injected = pd.DataFrame(fake_rows) if fake_rows else pd.DataFrame(columns=df.columns.tolist()+["viz_emotion","_is_visual_only"])
        if "viz_emotion" not in df.columns:
            df["viz_emotion"] = df["emotion"]
            df["_is_visual_only"] = False
        df_viz = pd.concat([df, df_injected], ignore_index=True)
    else:
        df["viz_emotion"] = df["emotion"]
        df["_is_visual_only"] = False
        df_viz = df

    # Density upsampling to target_points
    if len(df_viz) < target_points:
        need_more = target_points - len(df_viz)
        idxs = rng.integers(0, len(df_viz), size=need_more)
        extra = df_viz.iloc[idxs].copy()
        extra["_dup"] = True
        df_viz = pd.concat([df_viz, extra], ignore_index=True)

    # Shuffle for visual randomness
    df_viz = df_viz.sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)
    return df_viz

# -------------------------------------------------------
# Main render
# -------------------------------------------------------
def create_constellation(
    df_viz: pd.DataFrame,
    palette_name: str,
    layers: list,
    width: int,
    height: int,
    seed: int,
    size_scale: float,
    connect_k: int
):
    rng = np.random.default_rng(seed)

    # Background
    if "Background" in layers:
        bg = draw_background_gradient(width, height,
                                      top_rgb=(0.03, 0.05, 0.12),
                                      bottom_rgb=(0.00, 0.00, 0.00))
    else:
        bg = Image.new("RGB", (width, height), color=(0, 0, 0))

    # Matplotlib transparent canvas
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    cmap = PALETTES.get(palette_name, PALETTES["vivid"])

    n = len(df_viz)
    xs, ys = stratified_positions(n, rng)

    comp = df_viz["compound"].astype(float).to_numpy()
    intensity = np.clip(np.abs(comp), 0, 1)
    sizes = (16 + 230 * (intensity**0.85) * size_scale).tolist()
    alphas = (0.28 + 0.68 * intensity).tolist()

    # Map viz_emotion to colors
    colors = []
    for emo in df_viz["viz_emotion"].tolist():
        colors.append(cmap.get(emo, cmap["mixed"]))

    # Stardust for richness
    if "Stardust" in layers:
        num_dust = max(200, n // 3)
        dx = rng.uniform(0.0, 1.0, size=num_dust)
        dy = rng.uniform(0.0, 1.0, size=num_dust)
        ds = rng.uniform(2, 14, size=num_dust)
        da = rng.uniform(0.05, 0.16, size=num_dust)
        dc = []
        for _ in range(num_dust):
            dc.append((0.88, 0.90, 0.98) if rng.random() < 0.6 else (0.98, 0.90, 0.85))
        ax.scatter(dx, dy, s=ds, c=dc, alpha=da, linewidths=0, marker="o")

    # Glow underlay
    if "Glow" in layers and n > 0:
        ax.scatter(xs, ys, s=[s*3.2 for s in sizes], c=colors,
                   alpha=[a*0.18 for a in alphas], linewidths=0, marker="o")

    # Stars
    if "Stars" in layers and n > 0:
        ax.scatter(xs, ys, s=sizes, c=colors, alpha=alphas, linewidths=0, marker="o")

    # Connections
    if "Connections" in layers and n >= 3:
        pts = np.column_stack([xs, ys])
        for i in range(len(pts)):
            d = np.sum((pts - pts[i])**2, axis=1)
            nn = np.argsort(d)[1:1+max(1, connect_k)]
            for j in nn:
                ax.plot([pts[i,0], pts[j,0]],
                        [pts[i,1], pts[j,1]],
                        linewidth=0.35, alpha=0.14, c="white")

    # Export & composite with bloom
    buf = BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    fg = Image.open(buf).convert("RGBA")
    bloom = fg.filter(ImageFilter.GaussianBlur(radius=1.0))
    comp_img = Image.alpha_composite(bg.convert("RGBA"), bloom)
    comp_img = Image.alpha_composite(comp_img, fg)

    out = BytesIO()
    comp_img.save(out, format="PNG")
    out.seek(0)
    return out

# -------------------------------------------------------
# UI — Instructions
# -------------------------------------------------------
with st.expander("Instructions", expanded=False):
    st.markdown("""
    **How to use**
    1) Choose a data source (Upload CSV / Paste text / Fetch news).  
    2) Set palette, layers, and 'Fullness target' to control visual richness.  
    3) Adjust random seed, star size, and connection count.  
    4) View and download the constellation.  
    """)

# -------------------------------------------------------
# Sidebar — Data input
# -------------------------------------------------------
st.sidebar.header("Data input")
mode = st.sidebar.radio("Select data source:", ["Upload CSV", "Paste text", "Fetch news"], index=0)
df = pd.DataFrame()

if mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload a CSV (must contain 'text')", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.sidebar.error("Failed to read CSV. Check encoding and delimiter.")
elif mode == "Paste text":
    txt = st.sidebar.text_area("Paste multiple lines (one text per line):", height=220,
                               placeholder="I love the sunset.\nMarkets look uncertain today.\nFeeling proud of our team.")
    if st.sidebar.button("Add to dataset", use_container_width=True):
        rows = [t for t in txt.splitlines() if t.strip()]
        if rows:
            df = pd.DataFrame({"text": rows})
            df["timestamp"] = str(date.today())
elif mode == "Fetch news":
    kw = st.sidebar.text_input("Keyword (e.g., economy / technology / climate):", "technology")
    if st.sidebar.button("Fetch from NewsAPI", use_container_width=True):
        key = st.secrets.get("NEWS_API_KEY", "")
        if not key:
            st.sidebar.error("Missing NEWS_API_KEY. Add it in Streamlit Secrets.")
        else:
            df = fetch_news(key, keyword=kw)

if df.empty:
    # fallback sample
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("Using sample_data.csv (no data provided).")
    except Exception:
        df = pd.DataFrame({"text": [
            "I can't believe how beautiful the sky is tonight!",
            "The new update is fantastic and smooth.",
            "Why is it raining again? Feeling a bit low.",
            "Our team finally shipped the feature! Proud and grateful.",
            "Markets look volatile; investors are anxious.",
        ]})
        df["timestamp"] = str(date.today())

if "text" not in df.columns:
    st.error("The dataset must include a 'text' column.")
    st.stop()

# -------------------------------------------------------
# Sidebar — Visual options
# -------------------------------------------------------
st.sidebar.header("Emotion & Style")
palette_name = st.sidebar.selectbox("Palette:", ["vivid", "pastel"], index=0)
layer_options = ["Background", "Stars", "Glow", "Connections", "Stardust"]
selected_layers = st.sidebar.multiselect("Layers:", options=layer_options, default=layer_options)

seed = st.sidebar.number_input("Random seed:", min_value=0, max_value=2_000_000_000, value=42, step=1)
size_scale = st.sidebar.slider("Star size scale:", 0.5, 2.0, 1.0, 0.05)
connect_k = st.sidebar.slider("Connections per star:", 0, 4, 2, 1)

st.sidebar.markdown("---")
st.sidebar.header("Fullness & Filters")
min_emotions = st.sidebar.slider("Minimum distinct colors (visual):", 3, 12, 8, 1)
target_points = st.sidebar.slider("Fullness target points:", 200, 3000, 800, 50)
cmp_min, cmp_max = st.sidebar.slider("Compound range:", -1.0, 1.0, (-1.0, 1.0), 0.01)

# -------------------------------------------------------
# Sentiment + emotion mapping
# -------------------------------------------------------
with st.spinner("Analyzing sentiment and mapping emotions..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

# Filter by compound if requested
df = df[(df["compound"] >= cmp_min) & (df["compound"] <= cmp_max)].reset_index(drop=True)

# Prepare visualization DataFrame with enrichment & density balance
rng = np.random.default_rng(seed)
df_viz = enrich_and_balance(df, palette_name=palette_name, rng=rng,
                            min_emotions=min_emotions, target_points=target_points)

# Emotion filter AFTER enrichment
available_emotions = sorted(list(set(df_viz["viz_emotion"].unique().tolist()) | set(ALL_EMOTIONS)))
show_emotions = st.sidebar.multiselect("Show emotions:", options=available_emotions,
                                       default=sorted(df_viz["viz_emotion"].unique().tolist()))
df_viz = df_viz[df_viz["viz_emotion"].isin(show_emotions)].reset_index(drop=True)

# -------------------------------------------------------
# Draw & table
# -------------------------------------------------------
left, right = st.columns([0.58, 0.42])

with left:
    st.subheader("⭐ Constellation")
    if df_viz.empty:
        st.warning("No data points under current filters.")
    else:
        img_buf = create_constellation(
            df_viz=df_viz,
            palette_name=palette_name,
            layers=selected_layers,
            width=1600,
            height=900,
            seed=seed,
            size_scale=size_scale,
            connect_k=connect_k
        )
        st.image(img_buf, caption=f"Emotional Constellation — {palette_name} palette", use_column_width=True)
        st.download_button("💾 Download PNG", data=img_buf, file_name="emotional_constellation.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    cols = ["text", "emotion", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns: cols.insert(1, "timestamp")
    if "source" in df.columns: cols.insert(2, "source")
    st.dataframe(df[cols], use_container_width=True, height=480)

st.markdown("---")
st.caption("Made with ❤️ — Auto-enriched colors and density. Always full, always colorful. Data → Emotion → Generative Art → Streamlit © 2025")
