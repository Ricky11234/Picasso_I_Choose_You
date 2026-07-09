"""
Picasso, I Choose You! — SHOWCASE app.

A lightweight gallery/description page that deploys to Streamlit Cloud with no
heavy dependencies (only streamlit + pillow). The real, compute-heavy app
(App.py) runs locally. This page presents the project and links to the repo.

TO ADD YOUR OWN EXAMPLES:
Drop your images into the assets/ folder and update the EXAMPLES list below.
Each full example needs a content image, a style image, and your generated
output. Missing files simply show a placeholder, so the app never breaks.
"""

import os
import streamlit as st

# ---- EDIT THESE ------------------------------------------------------------
REPO_URL = "https://github.com/Ricky11234/Picasso_I_Choose_You"  # <-- your repo

# Add as many examples as you like. Put all image files in the assets/ folder.
# An entry with NO content/style pair (both None) renders its output on its own
# as a single featured image — handy while you only have one result to show.
EXAMPLES = [
    {
        # The one result you have so far, shown on its own as a featured image.
        # Once you generate a matching content+style pair, fill those in and it
        # will render as a full content → style → result triptych instead.
        "title": "Featured result",
        "content": None,
        "style":   None,
        "output":  "assets/example.png",
        "note":    "A sample output from the local style-transfer app. More coming soon!",
    },

    # ---- PLACEHOLDERS ------------------------------------------------------
    # Uncomment and fill these in as you generate more content/style/output
    # sets. Drop the files into assets/ and point each path at them. Any file
    # that doesn't exist yet just shows an "add me" prompt, so nothing breaks.
    # {
    #     "title": "Example 2",
    #     "content": "assets/ex2_content.jpg",
    #     "style":   "assets/ex2_style.jpg",
    #     "output":  "assets/ex2_output.png",
    #     "note":    "A short caption describing this result.",
    # },
    # {
    #     "title": "Example 3",
    #     "content": "assets/ex3_content.jpg",
    #     "style":   "assets/ex3_style.jpg",
    #     "output":  "assets/ex3_output.png",
    #     "note":    "Another caption.",
    # },
    # ------------------------------------------------------------------------
]
HERO_IMAGE = "assets/hero.png"   # optional big banner image; ignored if absent
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Picasso, I Choose You!", page_icon="🎨", layout="wide")

st.markdown(
    """
    <style>
    .stButton > button, .stLinkButton > a {
      background: linear-gradient(90deg,#E63946,#F4A261,#2A9D8F,#2E4BC6);
      background-size: 250% 100%; color:#fff !important; border:none; border-radius:12px;
      padding:0.6rem 1.4rem; font-weight:700; text-decoration:none;
      transition: background-position .5s ease, transform .1s ease;
    }
    .stButton > button:hover, .stLinkButton > a:hover {
      background-position:100% 0; transform:translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"➕ Add **{path}**")


# ---- Header ----
st.markdown(
    '<div style="background:linear-gradient(120deg,#E63946,#F4A261,#E9C46A,'
    '#2A9D8F,#2E4BC6,#7B2CBF);padding:1.6rem 1.8rem;border-radius:18px;'
    'color:#fff;box-shadow:0 6px 18px rgba(0,0,0,0.12);">'
    '<div style="font-size:2.1rem;font-weight:800;">🎨 Picasso, I Choose You!</div>'
    '<div style="opacity:0.96;margin-top:0.35rem;font-size:1.12rem;">'
    'A neural style transfer studio for designers and artists — repaint any photo '
    'in the style of any artwork.</div></div>',
    unsafe_allow_html=True,
)

if os.path.exists(HERO_IMAGE):
    st.image(HERO_IMAGE, use_container_width=True)

st.markdown(
    "\nBring a photo and a piece of art — a painting, a comic panel, a texture — and "
    "the tool blends them into something new for **posters, covers, social posts, "
    "prints, and mood boards**. Because the style is captured as texture and colour "
    "statistics rather than copied literally, every result is an original "
    "reinterpretation, not a one-click filter."
)

st.link_button("▶  Get the code & run it locally", REPO_URL)
st.caption("The full interactive app is compute-heavy (PyTorch), so it runs on your "
           "own machine. This page showcases what it produces.")

st.divider()

# ---- Gallery ----
st.header("Gallery")
st.caption("Content image  +  style image  →  generated result")
for ex in EXAMPLES:
    st.subheader(ex["title"])
    if ex.get("content") or ex.get("style"):
        # Full example: content + style + generated result, side by side.
        c1, c2, c3 = st.columns(3)
        with c1:
            show_img(ex["content"], "Content")
        with c2:
            show_img(ex["style"], "Style")
        with c3:
            show_img(ex["output"], "Result")
    else:
        # Featured single image (no content/style pair yet) — show it larger,
        # centered, so one result still looks intentional rather than empty.
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            show_img(ex["output"], "Result")
    if ex.get("note"):
        st.caption(ex["note"])
    st.write("")

st.divider()

# ---- How it works ----
st.header("How it works")
st.markdown(
    "This uses **neural style transfer** (Gatys, Ecker & Bethge, 2015). A frozen, "
    "pretrained **VGG19** network reads both images. The **content** of your photo is "
    "taken from a deep layer's activations; the **style** of the artwork is taken from "
    "the correlations between features (**Gram matrices**) across several layers. "
    "Starting from your photo (or random noise), the *pixels* of a generated image are "
    "optimised — using **L-BFGS with a strong-Wolfe line search** — until the content "
    "matches your photo and the style matches the artwork."
)

st.header("Tech stack")
st.markdown(
    "- **Python** · **PyTorch** — the deep learning engine\n"
    "- **VGG19** (pretrained, via torchvision) — a frozen feature extractor\n"
    "- **Gram matrices** — the style representation\n"
    "- **L-BFGS + strong-Wolfe line search** — optimises the image pixels\n"
    "- **Total Variation loss** — keeps output smooth\n"
    "- **Streamlit** — the interactive UI (and this showcase page)"
)

st.header("Run it yourself")
st.markdown(f"The full app lives on GitHub. Clone it and run locally:")
st.code(
    "git clone " + REPO_URL + "\n"
    "cd Picasso_I_Choose_You\n"
    "pip install torch torchvision streamlit pillow numpy\n"
    "streamlit run App.py",
    language="bash",
)

st.divider()
st.header("Credits")
st.markdown(
    "- Prof. Mitesh M. Khapra — *Deep Art* lecture, CS7015 (Deep Learning), IIT Madras\n"
    "- Leon A. Gatys, Alexander S. Ecker & Matthias Bethge — *A Neural Algorithm of "
    "Artistic Style* (2015), arXiv:1508.06576\n"
    "- VGG19 — Simonyan & Zisserman (2014)"
)