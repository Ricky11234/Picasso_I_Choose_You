"""
Picasso, I Choose You! — SHOWCASE page.

Lightweight (streamlit + pillow only) gallery/description page for Streamlit
Cloud. The real PyTorch app (App.py) runs locally. Add your generated images to
the assets/ folder and list them in EXAMPLES below.
"""

import os
import streamlit as st

# ---- EDIT THESE ------------------------------------------------------------
REPO_URL = "https://github.com/Ricky11234/Picasso_I_Choose_You"
FEATURED = "assets/example.png"          # a single hero result (shown if present)

# Add full sets as you generate them. Missing files just show a placeholder.
EXAMPLES = [
    # {"title": "Skyline × Starry Night",
    #  "content": "assets/ex1_content.jpg",
    #  "style":   "assets/ex1_style.jpg",
    #  "output":  "assets/ex1_output.png",
    #  "note":    "A short caption."},
]
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Picasso, I Choose You!", page_icon="🎨", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&display=swap');

    .pic-hero { display:flex; align-items:center; gap:1.1rem; margin:0.2rem 0 0.4rem; }
    .pic-emblem { flex:0 0 auto; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.14)); }
    .pic-wordmark {
      font-family:'Fredoka',sans-serif; font-weight:700; font-size:2.9rem; line-height:1.02;
      background:linear-gradient(90deg,#E63946,#F4A261,#E9C46A,#2A9D8F,#2E4BC6,#7B2CBF);
      background-size:200% auto; -webkit-background-clip:text; background-clip:text;
      -webkit-text-fill-color:transparent; color:transparent;
      animation: picshift 9s linear infinite;
    }
    @keyframes picshift { to { background-position:200% center; } }
    .pic-tag { font-family:'Fredoka',sans-serif; font-weight:500; color:#5b6472;
               font-size:1.15rem; margin-top:0.2rem; letter-spacing:.2px; }
    h1,h2,h3 { font-family:'Fredoka',sans-serif !important; }

    .stButton > button, .stLinkButton > a {
      background: linear-gradient(90deg,#E63946,#7B2CBF); color:#fff !important;
      border:none; border-radius:12px; padding:0.6rem 1.5rem; font-weight:700;
      font-family:'Fredoka',sans-serif; text-decoration:none; display:inline-block;
      box-shadow: 0 3px 12px rgba(123,44,191,0.22);
      transition: transform .1s ease, box-shadow .2s ease;
    }
    .stButton > button:hover, .stLinkButton > a:hover {
      transform:translateY(-1px); box-shadow: 0 6px 18px rgba(230,57,70,0.28);
    }
    @media (prefers-reduced-motion: reduce) { .pic-wordmark { animation:none; } }
    </style>
    """,
    unsafe_allow_html=True,
)

EMBLEM_SVG = (
    '<svg class="pic-emblem" viewBox="0 0 100 100" width="78" height="78" '
    'role="img" aria-label="Palette Ball">'
    '<defs><linearGradient id="pb" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#E63946"/><stop offset=".35" stop-color="#F4A261"/>'
    '<stop offset=".65" stop-color="#2A9D8F"/><stop offset="1" stop-color="#7B2CBF"/>'
    '</linearGradient></defs>'
    '<circle cx="50" cy="50" r="45" fill="#fff" stroke="#1F2933" stroke-width="5"/>'
    '<path d="M7 50a43 43 0 0 1 86 0Z" fill="url(#pb)"/>'
    '<circle cx="35" cy="33" r="3.6" fill="#fff" opacity=".9"/>'
    '<circle cx="53" cy="26" r="3.6" fill="#E9C46A"/>'
    '<circle cx="69" cy="35" r="3.6" fill="#2E4BC6"/>'
    '<rect x="5" y="46" width="90" height="8" fill="#1F2933"/>'
    '<circle cx="50" cy="50" r="14" fill="#fff" stroke="#1F2933" stroke-width="5"/>'
    '<circle cx="50" cy="50" r="6" fill="#E63946"/></svg>'
)


def show_img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"➕ Add **{path}**")


# ---- Hero ----
st.markdown(
    f'<div class="pic-hero">{EMBLEM_SVG}'
    '<div><div class="pic-wordmark">Picasso, I Choose You!</div>'
    '<div class="pic-tag">Gotta paint \'em all — a neural style transfer studio '
    'for designers &amp; artists.</div></div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    "Bring a photo and a piece of art — a painting, a comic panel, a texture — and "
    "the tool repaints your photo in that artwork's style. Every result is an original "
    "reinterpretation, not a one-click filter — great for **posters, covers, social "
    "posts, prints, and mood boards**."
)
st.link_button("▶  Get the code & run it locally", REPO_URL)
st.caption("The full interactive app is compute-heavy (PyTorch), so it runs on your own "
           "machine. This page shows what it makes.")

st.divider()

# ---- Gallery ----
st.header("🖼️ Gallery")
if os.path.exists(FEATURED):
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        with st.container(border=True):
            st.image(FEATURED, caption="Featured result", use_container_width=True)
elif not EXAMPLES:
    st.info("➕ Add a result to **assets/example.png** to feature it here.")

if EXAMPLES:
    st.caption("Content image  +  style image  →  generated result")
for ex in EXAMPLES:
    st.subheader(ex.get("title", ""))
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            show_img(ex["content"], "Content")
        with c2:
            show_img(ex["style"], "Style")
        with c3:
            show_img(ex["output"], "Result")
        if ex.get("note"):
            st.caption(ex["note"])

st.divider()

# ---- How it works ----
st.header("🧠 How it works")
st.markdown(
    "A frozen, pretrained **VGG19** network reads both images. The **content** of your "
    "photo comes from a deep layer's activations; the **style** of the artwork comes "
    "from the correlations between features (**Gram matrices**) across several layers. "
    "Starting from your photo (or random noise), the *pixels* of a generated image are "
    "optimised — with **L-BFGS + a strong-Wolfe line search** — until the content "
    "matches your photo and the style matches the artwork. Based on Gatys, Ecker & "
    "Bethge (2015)."
)

st.header("🛠️ Tech stack")
st.markdown(
    "- **Python · PyTorch** — the deep learning engine\n"
    "- **VGG19** (pretrained, via torchvision) — a frozen feature extractor\n"
    "- **Gram matrices** — the style representation\n"
    "- **L-BFGS + strong-Wolfe line search** — optimises the image pixels\n"
    "- **Total Variation loss** — keeps output smooth\n"
    "- **Streamlit** — the interactive UI (and this page)"
)

st.header("🚀 Run it yourself")
st.code(
    "git clone " + REPO_URL + "\n"
    "cd Picasso_I_Choose_You\n"
    "pip install torch torchvision streamlit pillow numpy\n"
    "streamlit run App.py",
    language="bash",
)

st.divider()
st.header("🙏 Credits")
st.markdown(
    "- Prof. Mitesh M. Khapra — *Deep Art* lecture, CS7015 (Deep Learning), IIT Madras\n"
    "- Leon A. Gatys, Alexander S. Ecker & Matthias Bethge — *A Neural Algorithm of "
    "Artistic Style* (2015), arXiv:1508.06576\n"
    "- VGG19 — Simonyan & Zisserman (2014)"
)