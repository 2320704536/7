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

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Constellation — Galaxy Edition", page_icon="✨", layout="wide")
st.title("🌌 Emotional Constellation — Galaxy Realistic Edition")
st.caption("English UI • Galaxy-like visuals • Layered rendering • Always colorful & full • Logical sidebar.")

# =========================
# Resources
# =========================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_vader()

# =========================
# NewsAPI
# =========================
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

# =========================
# Palettes & Themes
# =========================
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

THEMES = {
    # (center_rgb, edge_rgb, small_star_colors)
    "Galaxy Blue": ((0.06, 0.09, 0.20), (0.00, 0.00, 0.00), [(0.92,0.95,1.0), (0.80,0.86,1.0)]),
    "Warm Nebula": ((0.12, 0.06, 0.18), (0.00, 0.00, 0.00), [(1.00,0.92,0.88), (0.98,0.85,0.92)]),
    "Aurora Mist": ((0.05, 0.12, 0.12), (0.00, 0.00, 0.00), [(0.90,1.00,0.98), (0.85,0.95,1.00)]),
}

# =========================
# Sentiment & emotion mapping
# =========================
def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row) -> str:
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]
    if comp >= 0.7 and pos > 0.5:             return "joy"
    if comp >= 0.55 and pos > 0.45:           return "love"
    if comp >= 0.45 and pos > 0.40:           return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30:    return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5:    return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5:     return "surprise"
    if comp <= -0.65 and neg > 0.5:           return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45:  return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3:               return "anxiety"
    if neg > 0.45 and pos < 0.1:              return "disgust"
    if neu > 0.75 and abs(comp) < 0.1:        return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25:             return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15:    return "awe"
    return "neutral"

# =========================
# Visual helpers
# =========================
def to_uint8_rgb(rgb):
    return tuple(int(255 * max(0, min(1, c))) for c in rgb)

def draw_radial_gradient(width, height, center_rgb, edge_rgb):
    cx, cy = width/2, height/2
    y = np.arange(height).reshape(-1,1)
    x = np.arange(width).reshape(1,-1)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    dist = (dist / dist.max())
    dist = np.clip(dist, 0, 1)**1.25  # ease curve
    c = np.array(center_rgb).reshape(1,1,3)
    e = np.array(edge_rgb).reshape(1,1,3)
    img = (c*(1-dist[...,None]) + e*dist[...,None])
    return Image.fromarray((img*255).astype(np.uint8), mode="RGB")

def stratified_positions(n, rng):
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

# =========================
# Color enrichment & density balance
# =========================
def enrich_and_balance(df, palette_name, rng, min_emotions=8, target_points=900):
    df = df.copy()
    cmap = PALETTES.get(palette_name, PALETTES["vivid"])

    if "emotion" not in df.columns:
        raise ValueError("missing 'emotion' column before enrichment")

    present = df["emotion"].unique().tolist()
    need = max(0, min_emotions - len(present))
    if "viz_emotion" not in df.columns:
        df["viz_emotion"] = df["emotion"]
        df["_is_visual_only"] = False

    if need > 0:
        extra = [e for e in cmap.keys() if e not in present]
        rng.shuffle(extra)
        chosen = extra[:need] if need <= len(extra) else extra
        inject_n = max(8, len(df) // max(6, len(present) or 1))
        fake_rows = []
        for emo in chosen:
            pick = min(inject_n, len(df))
            idxs = rng.integers(0, len(df), size=pick)
            for i in idxs:
                row = df.iloc[i].to_dict()
                row["viz_emotion"] = emo
                row["_is_visual_only"] = True
                fake_rows.append(row)
        if fake_rows:
            df = pd.concat([df, pd.DataFrame(fake_rows)], ignore_index=True)

    if len(df) < target_points:
        need_more = target_points - len(df)
        idxs = rng.integers(0, len(df), size=need_more)
        extra = df.iloc[idxs].copy()
        extra["_dup"] = True
        df = pd.concat([df, extra], ignore_index=True)

    df = df.sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)
    return df

# =========================
# Main renderer
# =========================
def render_constellation(
    df_viz: pd.DataFrame,
    palette_name: str,
    theme_name: str,
    layers: list,
    width: int,
    height: int,
    seed: int,
    size_scale: float,
    connect_k: int
):
    rng = np.random.default_rng(seed)

    center_rgb, edge_rgb, small_colors = THEMES[theme_name]
    bg = draw_radial_gradient(width, height, center_rgb, edge_rgb) if "Background" in layers else Image.new("RGB", (width, height), (0,0,0))

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
    sizes = (14 + 220 * (intensity**0.85) * size_scale).tolist()
    alphas = (0.30 + 0.65 * intensity).tolist()

    main_colors = [cmap.get(emo, cmap["mixed"]) for emo in df_viz["viz_emotion"].tolist()]

    # Small stars (starfield)
    if "Small Stars" in layers:
        num_small = max(450, n)  # rich starfield
        s_x = rng.uniform(0.0, 1.0, size=num_small)
        s_y = rng.uniform(0.0, 1.0, size=num_small)
        s_s = rng.uniform(3, 16, size=num_small)
        s_a = rng.uniform(0.05, 0.22, size=num_small)
        s_c = [small_colors[0] if rng.random() < 0.6 else small_colors[1] for _ in range(num_small)]
        ax.scatter(s_x, s_y, s=s_s, c=s_c, alpha=s_a, linewidths=0, marker="o")

    # Glow layer
    if "Glow" in layers and n > 0:
        ax.scatter(xs, ys, s=[s*3.2 for s in sizes], c=main_colors, alpha=[a*0.18 for a in alphas], linewidths=0, marker="o")

    # Main stars
    if "Main Stars" in layers and n > 0:
        ax.scatter(xs, ys, s=sizes, c=main_colors, alpha=alphas, linewidths=0, marker="o")

    # Constellation lines (very subtle)
    if "Constellation Lines" in layers and n >= 3:
        pts = np.column_stack([xs, ys])
        for i in range(len(pts)):
            d = np.sum((pts - pts[i])**2, axis=1)
            nn = np.argsort(d)[1:1+max(1, connect_k)]
            for j in nn:
                ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]],
                        linewidth=0.35, alpha=0.12, c="white")

        # Export & composite with bloom
    buf = BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    fg = Image.open(buf).convert("RGBA")
    bloom = fg.filter(ImageFilter.GaussianBlur(radius=1.0))

    # 🔧 Ensure background and foreground have same size before blending
    bg_rgba = bg.convert("RGBA")
    if bg_rgba.size != fg.size:
        bg_rgba = bg_rgba.resize(fg.size, Image.LANCZOS)

    comp_img = Image.alpha_composite(bg_rgba, bloom)
    comp_img = Image.alpha_composite(comp_img, fg)

    out = BytesIO()
    comp_img.save(out, format="PNG")
    out.seek(0)
    return out

# =========================
# Sidebar — Logical sections
# =========================
with st.expander("Instructions", expanded=False):
    st.markdown("""
**How to use**
1) **Data Source**: choose where your texts come from.  
2) **Visualization Style**: choose theme, palette, and layers.  
3) **Emotion Mapping**: ensure color diversity and filter emotions.  
4) **Rendering Options**: randomness, size, density, and connections.  
5) **Output**: download or reset.  
""")

# ---- Section 1: Data Source
st.sidebar.header("1) Data Source")
mode = st.sidebar.radio("Select data source:", ["Upload CSV", "Paste Text", "Fetch News"], index=0)
df = pd.DataFrame()

if mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload a CSV (must contain 'text')", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.sidebar.error("Failed to read CSV. Check encoding and delimiter.")
elif mode == "Paste Text":
    txt = st.sidebar.text_area("Paste multiple lines (one text per line):", height=220,
                               placeholder="I love the sunset.\nMarkets look uncertain today.\nFeeling proud of our team.")
    if st.sidebar.button("Add to dataset", use_container_width=True):
        rows = [t for t in txt.splitlines() if t.strip()]
        if rows:
            df = pd.DataFrame({"text": rows})
            df["timestamp"] = str(date.today())
elif mode == "Fetch News":
    kw = st.sidebar.text_input("Keyword (e.g., economy / technology / climate):", "technology")
    if st.sidebar.button("Fetch from NewsAPI", use_container_width=True):
        key = st.secrets.get("NEWS_API_KEY", "")
        if not key:
            st.sidebar.error("Missing NEWS_API_KEY. Add it in Streamlit Secrets.")
        else:
            df = fetch_news(key, keyword=kw)

if df.empty:
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

# ---- Emotion analysis
with st.spinner("Analyzing sentiment and mapping emotions..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

# ---- Section 2: Visualization Style
st.sidebar.header("2) Visualization Style")
theme_name = st.sidebar.selectbox("Theme:", list(THEMES.keys()), index=0)
palette_name = st.sidebar.selectbox("Color palette:", ["vivid", "pastel"], index=0)
layer_options = ["Background", "Small Stars", "Main Stars", "Glow", "Constellation Lines"]
selected_layers = st.sidebar.multiselect("Layers:", options=layer_options, default=layer_options)

# ---- Section 3: Emotion Mapping
st.sidebar.header("3) Emotion Mapping")
min_emotions = st.sidebar.slider("Minimum distinct colors (visual only):", 3, 12, 8, 1)
available_emotions = sorted(df["emotion"].unique().tolist())
show_emotions = st.sidebar.multiselect("Show emotions:", options=ALL_EMOTIONS, default=available_emotions)
cmp_min, cmp_max = st.sidebar.slider("Compound range:", -1.0, 1.0, (-1.0, 1.0), 0.01)

df = df[(df["emotion"].isin(show_emotions)) & (df["compound"] >= cmp_min) & (df["compound"] <= cmp_max)].reset_index(drop=True)

# ---- Section 4: Rendering Options
st.sidebar.header("4) Rendering Options")
seed = st.sidebar.number_input("Random seed:", min_value=0, max_value=2_000_000_000, value=42, step=1)
size_scale = st.sidebar.slider("Main star size scale:", 0.5, 2.0, 1.0, 0.05)
target_points = st.sidebar.slider("Fullness target points:", 300, 3000, 900, 50)
connect_k = st.sidebar.slider("Connections per star:", 0, 4, 2, 1)

# Create visualization dataframe with enrichment & density balance
rng = np.random.default_rng(seed)
df_viz = enrich_and_balance(df, palette_name=palette_name, rng=rng,
                            min_emotions=min_emotions, target_points=target_points)

# ---- Section 5: Output
st.sidebar.header("5) Output")
if st.sidebar.button("Reset all settings"):
    st.session_state.clear()
    st.experimental_rerun()

# =========================
# Draw & table
# =========================
left, right = st.columns([0.58, 0.42])

with left:
    st.subheader("⭐ Constellation")
    if df_viz.empty:
        st.warning("No data points under current filters.")
    else:
        img_buf = render_constellation(
            df_viz=df_viz,
            palette_name=palette_name,
            theme_name=theme_name,
            layers=selected_layers,
            width=1600,
            height=900,
            seed=seed,
            size_scale=size_scale,
            connect_k=connect_k
        )
        st.image(img_buf, caption=f"Emotional Constellation — {theme_name} • {palette_name}", use_column_width=True)
        st.download_button("💾 Download PNG", data=img_buf, file_name="emotional_constellation.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    cols = ["text", "emotion", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns: cols.insert(1, "timestamp")
    if "source" in df.columns: cols.insert(2, "source")
    st.dataframe(df[cols], use_container_width=True, height=520)

st.markdown("---")
st.caption("Made with ❤️ — Galaxy-like aesthetics with logical controls. Always colorful, always full. © 2025")
