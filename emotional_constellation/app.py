import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Constellation — Wang Xinru — Final Project", page_icon="✨", layout="wide")
st.title("🌌 Emotional Constellation — Wang Xinru — Final Project")

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
    params = {"q": keyword, "language": "en", "sortBy": "publishedAt", "pageSize": page_size, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        rows = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            desc = a.get("description") or ""
            txt = (title + " - " + desc).strip(" -")
            rows.append({"timestamp": (a.get("publishedAt") or "")[:10], "text": txt, "source": (a.get("source") or {}).get("name", "")})
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# =========================
# Built-in fixed colors (one per emotion) — RGB 0..255
# =========================
DEFAULT_RGB = {
    "joy":        (255, 215, 0),    # Gold
    "love":       (255, 105, 180),  # Hot Pink
    "pride":      (238, 130, 238),  # Violet
    "hope":       (50, 205, 50),    # Lime Green
    "curiosity":  (64, 224, 208),   # Turquoise
    "calm":       (135, 206, 235),  # Sky Blue
    "surprise":   (255, 165, 0),    # Orange
    "neutral":    (192, 192, 192),  # Silver
    "sadness":    (70, 130, 180),   # Steel Blue
    "anger":      (255, 69, 0),     # Orange Red
    "fear":       (147, 112, 219),  # Medium Purple
    "disgust":    (107, 142, 35),   # Olive Drab
    "anxiety":    (218, 165, 32),   # Goldenrod
    "boredom":    (169, 169, 169),  # Dark Gray
    "nostalgia":  (255, 218, 185),  # Peach Puff
    "gratitude":  (127, 255, 212),  # Aquamarine
    "awe":        (224, 255, 255),  # Light Cyan
    "trust":      (60, 179, 113),   # Medium Sea Green
    "confusion":  (221, 160, 221),  # Plum
    "mixed":      (0, 255, 170),    # Teal-ish
}
ALL_EMOTIONS = list(DEFAULT_RGB.keys())

# English color names for built-ins (customs will show as "Custom r,g,b")
COLOR_NAMES = {
    "joy":"Gold","love":"Hot Pink","pride":"Violet","hope":"Lime Green","curiosity":"Turquoise","calm":"Sky Blue",
    "surprise":"Orange","neutral":"Silver","sadness":"Steel Blue","anger":"Orange Red","fear":"Medium Purple","disgust":"Olive Drab",
    "anxiety":"Goldenrod","boredom":"Dark Gray","nostalgia":"Peach Puff","gratitude":"Aquamarine","awe":"Light Cyan","trust":"Medium Sea Green",
    "confusion":"Plum","mixed":"Teal"
}

# =========================
# Themes (radial background)
# =========================
THEMES = {
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
def draw_radial_gradient(width, height, center_rgb, edge_rgb):
    cx, cy = width/2, height/2
    y = np.arange(height).reshape(-1,1)
    x = np.arange(width).reshape(1,-1)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    dist = (dist / dist.max())
    dist = np.clip(dist, 0, 1)**1.25
    c = np.array(center_rgb).reshape(1,1,3)
    e = np.array(edge_rgb).reshape(1,1,3)
    img = (c*(1-dist[...,None]) + e*dist[...,None])
    return Image.fromarray((img*255).astype(np.uint8), mode="RGB")

def stratified_positions(n, rng):
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    xs, ys = [], []
    for r in range(rows):
        for c in range(cols):
            if len(xs) >= n: break
            x0 = (c + rng.uniform(0.15, 0.85)) / cols
            y0 = (r + rng.uniform(0.15, 0.85)) / rows
            xs.append(np.clip(x0, 0.02, 0.98)); ys.append(np.clip(y0, 0.06, 0.94))
    return np.array(xs), np.array(ys)

# =========================
# Palette state (CSV + custom)
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state: st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state: st.session_state["custom_palette"] = {}  # {emotion:(r,g,b)}

def get_active_palette():
    """Return emotion -> (r,g,b) based on CSV/custom toggle."""
    if st.session_state.get("use_csv_palette") and st.session_state.get("custom_palette"):
        return dict(st.session_state["custom_palette"])
    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged

def add_custom_emotion(emotion: str, r: int, g: int, b: int):
    if not emotion:
        st.warning("Emotion name cannot be empty.")
        return
    r = int(np.clip(r, 0, 255)); g = int(np.clip(g, 0, 255)); b = int(np.clip(b, 0, 255))
    st.session_state["custom_palette"][emotion.strip()] = (r, g, b)

def import_palette_csv(file):
    """CSV columns: emotion,r,g,b"""
    try:
        dfc = pd.read_csv(file)
        cols_lower = [c.lower() for c in dfc.columns]
        needed = {"emotion","r","g","b"}
        if not needed.issubset(set(cols_lower)):
            st.error("CSV must include columns: emotion, r, g, b")
            return
        colmap = {c.lower(): c for c in dfc.columns}
        em = colmap["emotion"]; rc = colmap["r"]; gc = colmap["g"]; bc = colmap["b"]
        pal = {}
        for _, row in dfc.iterrows():
            emo = str(row[em]).strip()
            try:
                r = int(row[rc]); g = int(row[gc]); b = int(row[bc])
            except Exception:
                continue
            r = int(np.clip(r,0,255)); g = int(np.clip(g,0,255)); b = int(np.clip(b,0,255))
            if emo: pal[emo] = (r,g,b)
        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV.")
    except Exception as e:
        st.error(f"Failed to import CSV: {e}")

def export_palette_csv(palette_dict: dict) -> BytesIO:
    dfp = pd.DataFrame([{"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in palette_dict.items()])
    buf = BytesIO(); dfp.to_csv(buf, index=False); buf.seek(0); return buf

# =========================
# Enrichment & density (visual only)
# =========================
def enrich_and_balance(df, active_palette, rng, min_emotions=6, target_points=450):
    df = df.copy()

    if df.empty or len(df) == 0:
        st.warning("No valid data available for visualization.")
        return pd.DataFrame(columns=["viz_emotion","compound"])

    if "emotion" not in df.columns:
        raise ValueError("missing 'emotion' column before enrichment")

    present = df["emotion"].unique().tolist()
    palette_keys = list(active_palette.keys())

    if "viz_emotion" not in df.columns:
        df["viz_emotion"] = df["emotion"]
        df["_is_visual_only"] = False

    # Ensure at least min_emotions distinct colors visually
    need = max(0, min_emotions - len(set(present)))
    if need > 0:
        extra = [e for e in palette_keys if e not in present]
        rng.shuffle(extra)
        chosen = extra[:need] if need <= len(extra) else extra
        inject_n = max(6, len(df) // max(6, len(present) or 1))
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

    # Density upsample to target_points (sparser default)
    if len(df) < target_points and len(df) > 0:
        need_more = target_points - len(df)
        idxs = rng.integers(0, len(df), size=need_more)
        extra = df.iloc[idxs].copy()
        extra["_dup"] = True
        df = pd.concat([df, extra], ignore_index=True)

    df = df.sample(frac=1.0, random_state=rng.integers(0,1_000_000)).reset_index(drop=True)
    return df

# =========================
# Renderer
# =========================
def render_constellation(df_viz: pd.DataFrame, active_palette: dict, theme_name: str, layers: list,
                         width: int, height: int, seed: int, size_scale: float, connect_k: int, starfield_factor: float):
    rng = np.random.default_rng(seed)
    center_rgb, edge_rgb, small_colors = THEMES[theme_name]
    bg = draw_radial_gradient(width, height, center_rgb, edge_rgb) if "Background" in layers else Image.new("RGB",(width,height),(0,0,0))

    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); fig.patch.set_alpha(0); ax.patch.set_alpha(0)

    n = len(df_viz)
    xs, ys = stratified_positions(n, rng)

    comp = df_viz["compound"].astype(float).to_numpy()
    intensity = np.clip(np.abs(comp), 0, 1)
    sizes = (12 + 200 * (intensity**0.85) * size_scale).tolist()
    alphas = (0.28 + 0.62 * intensity).tolist()

    # colors from active palette — one fixed color per emotion
    main_colors = []
    for emo in df_viz["viz_emotion"].tolist():
        rgb = active_palette.get(emo, active_palette.get("mixed", (0,255,170)))
        main_colors.append(tuple(c/255.0 for c in rgb))

    # Small starfield (sparser, tied to starfield_factor)
    if "Small Stars" in layers:
        base_small = int(np.clip(300 * starfield_factor, 120, 600))
        num_small = max(base_small, int(n * 0.4))
        s_x = rng.uniform(0.0,1.0,size=num_small)
        s_y = rng.uniform(0.0,1.0,size=num_small)
        s_s = rng.uniform(2,12,size=num_small)
        s_a = rng.uniform(0.05,0.18,size=num_small)
        s_c = [small_colors[0] if rng.random()<0.65 else small_colors[1] for _ in range(num_small)]
        ax.scatter(s_x, s_y, s=s_s, c=s_c, alpha=s_a, linewidths=0, marker="o")

    # Glow
    if "Glow" in layers and n>0:
        ax.scatter(xs, ys, s=[s*3.0 for s in sizes], c=main_colors, alpha=[a*0.16 for a in alphas], linewidths=0, marker="o")

    # Main stars
    if "Main Stars" in layers and n>0:
        ax.scatter(xs, ys, s=sizes, c=main_colors, alpha=alphas, linewidths=0, marker="o")

    # Constellation lines
    if "Constellation Lines" in layers and n>=3:
        pts = np.column_stack([xs, ys])
        for i in range(len(pts)):
            d = np.sum((pts - pts[i])**2, axis=1)
            nn = np.argsort(d)[1:1+max(1, connect_k)]
            for j in nn:
                ax.plot([pts[i,0], pts[j,0]],[pts[i,1], pts[j,1]], linewidth=0.35, alpha=0.10, c="white")

    # Composite & bloom (size-safe)
    buf = BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    fg = Image.open(buf).convert("RGBA")
    bloom = fg.filter(ImageFilter.GaussianBlur(radius=1.0))
    bg_rgba = bg.convert("RGBA")
    if bg_rgba.size != fg.size:
        bg_rgba = bg_rgba.resize(fg.size, Image.LANCZOS)
    comp_img = Image.alpha_composite(bg_rgba, bloom)
    comp_img = Image.alpha_composite(comp_img, fg)

    out = BytesIO(); comp_img.save(out, format="PNG"); out.seek(0); return out

# =========================
# Sidebar — Logical sections
# =========================
with st.expander("Instructions", expanded=False):
    st.markdown("""
**How to use**
1) **Data Source**: fetch texts via NewsAPI.  
2) **Visualization Style**: theme, layers.  
3) **Emotion Mapping**: filter emotions and compound range.  
4) **Custom Palette (RGB)**: add emotions with RGB, or import/export CSV.  
5) **Rendering Options**: randomness, size, star density, connections.  
6) **Output**: download or reset.  
""")

# ---- 1) Data Source (NEWS ONLY)
st.sidebar.header("1) Data Source (NewsAPI only)")
kw = st.sidebar.text_input("Keyword (e.g., economy / technology / climate):", "technology")
news_btn = st.sidebar.button("Fetch from NewsAPI", use_container_width=True)

df = pd.DataFrame()
if news_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY. Add it in Streamlit Secrets.")
    else:
        df = fetch_news(key, keyword=kw)

# Fallback sample so app can render before first fetch
if df.empty:
    df = pd.DataFrame({"text":[
        "I can't believe how beautiful the sky is tonight!",
        "The new update is fantastic and smooth.",
        "Why is it raining again? Feeling a bit low.",
        "Our team finally shipped the feature! Proud and grateful.",
        "Markets look volatile; investors are anxious.",
    ]})
    df["timestamp"] = str(date.today())

# Ensure dataset has 'text'
if "text" not in df.columns:
    st.error("The dataset must include a 'text' column.")
    st.stop()

# Sentiment + emotion
with st.spinner("Analyzing sentiment and mapping emotions..."):
    sentiments = df["text"].fillna("").apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sentiments.reset_index(drop=True)], axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

# ---- 2) Visualization Style
st.sidebar.header("2) Visualization Style")
theme_name = st.sidebar.selectbox("Theme:", list(THEMES.keys()), index=0)
layer_options = ["Background", "Small Stars", "Main Stars", "Glow", "Constellation Lines"]
selected_layers = st.sidebar.multiselect("Layers:", options=layer_options, default=layer_options)

# ---- 3) Emotion Mapping
st.sidebar.header("3) Emotion Mapping")
cmp_min, cmp_max = st.sidebar.slider("Compound range:", -1.0, 1.0, (-1.0, 1.0), 0.01)
available_emotions = sorted(df["emotion"].unique().tolist())

# Palette state before building selectors
init_palette_state()

# ---- 4) Custom Palette (RGB)
st.sidebar.header("4) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox("Use CSV palette (RGB editor)", value=st.session_state["use_csv_palette"])
st.session_state["use_csv_palette"] = use_csv

with st.sidebar.expander("Add Custom Emotion (RGB 0–255)", expanded=False):
    c1, c2, c3, c4 = st.columns([1.8, 1, 1, 1])
    emo_name = c1.text_input("Emotion name")
    r = c2.number_input("R (0–255)", 0, 255, 255, 1)
    g = c3.number_input("G (0–255)", 0, 255, 255, 1)
    b = c4.number_input("B (0–255)", 0, 255, 255, 1)
    if st.button("Add Emotion", use_container_width=True):
        add_custom_emotion(emo_name, r, g, b)
        st.success(f"Added: {emo_name} = ({r},{g},{b})")
    # Live list of all custom items
    custom_now = st.session_state.get("custom_palette", {})
    if custom_now:
        df_custom = pd.DataFrame([{"emotion": k, "r": v[0], "g": v[1], "b": v[2]} for k, v in sorted(custom_now.items())])
        st.dataframe(df_custom, use_container_width=True, height=200)
    else:
        st.caption("No custom colors yet.")

with st.sidebar.expander("Edit Palette / Import & Export CSV", expanded=False):
    upcsv = st.file_uploader("Import palette CSV (emotion,r,g,b)", type=["csv"])
    if upcsv is not None:
        import_palette_csv(upcsv)
    # Visible palette (merged or only CSV depending on toggle)
    current_pal = dict(DEFAULT_RGB)
    current_pal.update(st.session_state.get("custom_palette", {}))
    if st.session_state.get("use_csv_palette"):
        current_pal = dict(st.session_state.get("custom_palette", {}))
    if current_pal:
        pal_df = pd.DataFrame([{"emotion":k, "r":v[0], "g":v[1], "b":v[2]} for k,v in sorted(current_pal.items())])
        st.dataframe(pal_df, use_container_width=True, height=220)
        dl = export_palette_csv(current_pal)
        st.download_button("Download CSV", data=dl, file_name="palette.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No colors yet. Add emotions above or import a CSV.")

# Activate palette (after user actions)
ACTIVE_PALETTE = get_active_palette()

def emotion_label_with_name(e: str, pal: dict) -> str:
    """Return 'emotion (EnglishColorName)' or 'emotion (Custom r,g,b)'."""
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    rgb = pal.get(e, (0, 0, 0))
    return f"{e} (Custom {rgb[0]},{rgb[1]},{rgb[2]})"

# Emotion selector with English color names
final_labels_options = [emotion_label_with_name(e, ACTIVE_PALETTE) for e in ALL_EMOTIONS]
final_labels_default = [emotion_label_with_name(e, ACTIVE_PALETTE) for e in available_emotions]
selected_labels = st.sidebar.multiselect("Show emotions:", options=final_labels_options, default=final_labels_default)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

# filter df with selection and compound range
df = df[(df["emotion"].isin(selected_emotions)) & (df["compound"] >= cmp_min) & (df["compound"] <= cmp_max)].reset_index(drop=True)

# ---- 5) Rendering Options
st.sidebar.header("5) Rendering Options")
seed = st.sidebar.number_input("Random seed:", min_value=0, max_value=2_000_000_000, value=42, step=1)
size_scale = st.sidebar.slider("Main star size scale:", 0.5, 2.0, 1.0, 0.05)
target_points = st.sidebar.slider("Star Density (points):", 50, 1000, 450, 25)
connect_k = st.sidebar.slider("Connections per star:", 0, 4, 2, 1)

# Prepare visualization df (enrichment & density)
rng = np.random.default_rng(seed)
df_viz = enrich_and_balance(df, active_palette=ACTIVE_PALETTE, rng=rng, min_emotions=6, target_points=target_points)

# ---- 6) Output
st.sidebar.header("6) Output")
if st.sidebar.button("Reset all settings"):
    st.session_state.clear(); st.rerun()

# =========================
# Draw & table
# =========================
left, right = st.columns([0.58, 0.42])
with left:
    st.subheader("⭐ Constellation")
    if df_viz.empty:
        st.warning("No data points under current filters.")
    else:
        starfield_factor = np.clip(target_points / 450.0, 0.6, 2.0)
        img_buf = render_constellation(
            df_viz=df_viz, active_palette=ACTIVE_PALETTE, theme_name=theme_name, layers=selected_layers,
            width=1600, height=900, seed=seed, size_scale=size_scale, connect_k=connect_k, starfield_factor=starfield_factor
        )
        st.image(img_buf, caption=f"Emotional Constellation — {theme_name}", use_column_width=True)
        st.download_button("💾 Download PNG", data=img_buf, file_name="emotional_constellation.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    df_show = df.copy()
    def label_for_table(e):
        if e in COLOR_NAMES: return f"{e} ({COLOR_NAMES[e]})"
        rgb = ACTIVE_PALETTE.get(e, (0,0,0))
        return f"{e} (Custom {rgb[0]},{rgb[1]},{rgb[2]})"
    df_show["emotion_label"] = df_show["emotion"].apply(label_for_table)
    cols = ["text", "emotion_label", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns: cols.insert(1, "timestamp")
    if "source" in df.columns: cols.insert(2, "source")
    st.dataframe(df_show[cols], use_container_width=True, height=520)

st.markdown("---")
st.caption("© 2025")
