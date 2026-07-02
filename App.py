"""
Picasso, I Choose You! — Neural Style Transfer as a design tool.

Upload a photo and an artwork; the app repaints your photo in the artwork's
style. Built on the method from Gatys, Ecker & Bethge (2015), following
Prof. Mitesh Khapra's "Deep Art" lecture (CS7015, IIT Madras).

The heavy lifting (VGG feature extraction, Gram-based style loss, and the
pixel-optimisation loop) lives in the "ENGINE" section. The "INTERFACE"
section wraps it in controls an artist can use without knowing the maths.
"""

import io
import gc

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import streamlit as st

# ----------------------------------------------------------------------------
# ENGINE  (the model we built step by step — unchanged in spirit)
# ----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Layer index in vgg.features -> paper name. Only the six we use.
LAYER_NAMES = {0: "conv1_1", 5: "conv2_1", 10: "conv3_1",
               19: "conv4_1", 21: "conv4_2", 28: "conv5_1"}
CONTENT_LAYERS = ["conv4_2"]
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
_LAST_LAYER = max(LAYER_NAMES)  # stop the forward pass once we've grabbed everything


@st.cache_resource(show_spinner=False)
def load_vgg():
    """Load VGG19 once and reuse it across runs (frozen; weights never train)."""
    vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg.to(DEVICE)


def preprocess(pil_image, max_size):
    """PIL image -> normalised (1,3,H,W) tensor on DEVICE."""
    image = pil_image.convert("RGB")
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def deprocess(tensor):
    """Normalised tensor -> viewable PIL image."""
    image = tensor.clone().detach().cpu().squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)


def get_features(image, model):
    """Run the image through VGG, snapshotting activations at our chosen layers."""
    features = {}
    x = image
    for index, layer in enumerate(model):
        x = layer(x)
        if index in LAYER_NAMES:
            features[LAYER_NAMES[index]] = x
        if index == _LAST_LAYER:
            break
    return features


def gram_matrix(feature_map):
    """(1,C,H,W) -> normalised (C,C) Gram matrix (the style fingerprint)."""
    _, C, H, W = feature_map.size()
    V = feature_map.view(C, H * W)
    G = torch.mm(V, V.t())
    return G / (C * H * W)


def content_loss(content_feats, gen_feats):
    loss = 0.0
    for layer in CONTENT_LAYERS:
        loss = loss + torch.sum((content_feats[layer] - gen_feats[layer]) ** 2)
    return loss


def style_loss(style_grams, gen_feats, style_weights):
    loss = 0.0
    for layer in STYLE_LAYERS:
        A = gram_matrix(gen_feats[layer])
        E = torch.sum((style_grams[layer] - A) ** 2)
        loss = loss + style_weights[layer] * E
    return loss


def compute_style_weights(tilt):
    """
    Turn a single 'texture scale' dial into per-layer style weights.
    tilt < 0  -> favour shallow layers (fine, delicate texture)
    tilt = 0  -> equal weighting
    tilt > 0  -> favour deep layers (bold, large-scale shapes)
    """
    idx = np.arange(len(STYLE_LAYERS))          # 0 = shallow ... 4 = deep
    raw = np.exp(tilt * (idx - 2.0))
    raw = raw / raw.sum()
    return {layer: float(w) for layer, w in zip(STYLE_LAYERS, raw)}


def total_variation_loss(img):
    """
    Penalise pixel-to-pixel jaggedness so the result is smooth, not noisy.
    Sums the absolute differences between each pixel and its right/bottom neighbour.
    """
    return (torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
            torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))


def run_style_transfer(content_t, style_t, vgg, beta, steps, style_weights,
                       alpha=1.0, tv_weight=1.0, progress_cb=None):
    """Optimise the PIXELS of a generated image toward content + style. Returns a tensor."""
    content_feats = get_features(content_t, vgg)
    style_feats = get_features(style_t, vgg)
    style_grams = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}

    generated = content_t.clone().requires_grad_(True)
    # strong_wolfe line search makes each L-BFGS step actually find a good step
    # size, instead of a fixed lr=1 that stalls. This is what lets the style
    # fully "set in". max_iter=1 keeps 1 step == 1 progress-bar tick.
    optimizer = optim.LBFGS([generated], max_iter=1, line_search_fn="strong_wolfe")

    for step in range(steps):
        def closure():
            optimizer.zero_grad()
            gen_feats = get_features(generated, vgg)
            c = content_loss(content_feats, gen_feats)
            s = style_loss(style_grams, gen_feats, style_weights)
            tv = total_variation_loss(generated)
            total = alpha * c + beta * s + tv_weight * tv
            total.backward()
            return total

        optimizer.step(closure)
        if progress_cb is not None:
            progress_cb(step + 1, steps)

    return generated.detach()


# ----------------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Picasso, I Choose You!",
                   page_icon="🎨", layout="wide")

# --- a little colour, injected once (a painter's palette) --------------------
st.markdown(
    """
    <style>
    :root {
      --crimson:#E63946; --tangerine:#F4A261; --gold:#E9C46A;
      --teal:#2A9D8F; --ultramarine:#2E4BC6; --violet:#7B2CBF;
    }
    /* vibrant action buttons */
    .stButton > button, .stDownloadButton > button {
      background: linear-gradient(90deg,#E63946,#F4A261,#2A9D8F,#2E4BC6);
      background-size: 250% 100%;
      color:#fff; border:none; border-radius:12px;
      padding:0.6rem 1.4rem; font-weight:700; font-size:1.02rem;
      transition: background-position .5s ease, transform .1s ease, box-shadow .2s ease;
      box-shadow: 0 3px 10px rgba(46,75,198,0.18);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      background-position:100% 0; transform:translateY(-1px); color:#fff;
      box-shadow: 0 6px 16px rgba(230,57,70,0.25);
    }
    .stButton > button:disabled { background:#d9d4c9; color:#8f8a7c; box-shadow:none; }
    /* tinted upload dropzones */
    [data-testid="stFileUploaderDropzone"] {
      border:2px dashed var(--ultramarine);
      background: rgba(46,75,198,0.05);
      border-radius:14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def step_badge(num, text, color):
    """A numbered, coloured section heading."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;'
        f'margin:0.9rem 0 0.4rem;">'
        f'<span style="background:{color};color:#fff;width:1.9rem;height:1.9rem;'
        f'border-radius:50%;display:inline-flex;align-items:center;'
        f'justify-content:center;font-weight:800;flex:0 0 auto;">{num}</span>'
        f'<span style="font-size:1.3rem;font-weight:800;color:#1F2933;">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def chip(text, color):
    """A small coloured pill for live feedback under a control."""
    st.markdown(
        f'<div style="display:inline-block;background:{color}22;color:{color};'
        f'border:1px solid {color}66;padding:0.3rem 0.75rem;border-radius:999px;'
        f'font-size:0.86rem;font-weight:600;margin:0.15rem 0 0.2rem;">{text}</div>',
        unsafe_allow_html=True,
    )


st.title("🎨 Picasso, I Choose You!")
st.markdown(
    "**Repaint your photo in the style of any artwork.** "
    "Bring a photo and a piece of art — a painting, a comic panel, a texture — "
    "and blend them into something new for your posters, covers, and prints."
)

create_tab, about_tab = st.tabs(["Create", "How it works"])

# ---------- CREATE TAB -------------------------------------------------------
with create_tab:
    st.markdown(
        '<div style="background:linear-gradient(120deg,#E63946,#F4A261,#E9C46A,'
        '#2A9D8F,#2E4BC6,#7B2CBF);padding:1.3rem 1.6rem;border-radius:18px;'
        'margin:0.3rem 0 0.6rem;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,0.12);">'
        '<div style="font-size:1.55rem;font-weight:800;">Turn a photo into a painting</div>'
        '<div style="opacity:0.95;margin-top:0.25rem;font-size:1.02rem;">'
        'Upload a photo and an artwork, choose the vibe, and let the studio blend them.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    step_badge(1, "Bring your two images", "#E63946")

    col_content, col_style = st.columns(2)
    with col_content:
        content_file = st.file_uploader(
            "Your photo (the subject to keep)",
            type=["jpg", "jpeg", "png"], key="content")
        if content_file:
            content_img = Image.open(content_file)
            st.image(content_img, caption="Photo", use_container_width=True)

    with col_style:
        style_file = st.file_uploader(
            "The artwork (the look to borrow)",
            type=["jpg", "jpeg", "png"], key="style")
        if style_file:
            style_img = Image.open(style_file)
            st.image(style_img, caption="Style", use_container_width=True)

    step_badge(2, "Dial in the look", "#2A9D8F")

    # --- Style strength -> beta (log scale). strength 7 == the validated 1e7 ---
    strength = st.slider(
        "Style strength", min_value=1, max_value=10, value=7,
        help="How strongly the artwork's look takes over your photo.")
    if strength <= 3:
        chip("Gentle — your photo stays clearly recognisable", "#2A9D8F")
    elif strength <= 7:
        chip("Balanced — a clear repaint that still reads as your photo", "#2E4BC6")
    else:
        chip("Bold — the artwork's colours and textures take over", "#E63946")
    beta = 10.0 ** (5.0 + (strength - 1) / 3.0)

    col_a, col_b = st.columns(2)
    with col_a:
        size = st.slider(
            "Detail", min_value=256, max_value=512, value=400, step=32,
            help="Bigger = sharper, more detailed — but slower to create.")
        st.caption("Larger sizes look crisper but take longer. On the free cloud, "
                   "keep this around 400 or below to stay within memory.")
    with col_b:
        steps = st.slider(
            "Refinement passes", min_value=50, max_value=800, value=300, step=25,
            help="More passes let the style fully set in; fewer finish faster.")
        st.caption("Roughly linear with wait time. 300 gives a strong result; "
                   "drop to ~100 for a quick preview, raise toward 600+ for max effect.")

    with st.expander("Advanced — texture scale"):
        tilt = st.slider(
            "Fine  ←→  Bold", min_value=-1.5, max_value=1.5, value=0.0, step=0.1,
            help="Which scale of the artwork's texture to emphasise.")
        if tilt < -0.3:
            chip("Fine detail — delicate, brush-like grain", "#7B2CBF")
        elif tilt > 0.3:
            chip("Bold shapes — large sweeps and blocks of colour", "#F4A261")
        else:
            chip("Balanced — fine detail and large shapes", "#2A9D8F")
    style_weights = compute_style_weights(tilt)

    step_badge(3, "Create", "#7B2CBF")
    st.caption("This runs an optimisation on the free CPU cloud, so expect a few "
               "minutes. Lower the detail and passes for a quicker preview.")

    ready = content_file is not None and style_file is not None
    if not ready:
        st.info("Add both a photo and an artwork above to begin.")

    if st.button("Create artwork", type="primary", disabled=not ready):
        vgg = load_vgg()
        content_t = preprocess(Image.open(content_file), size)
        style_t = preprocess(Image.open(style_file), size)

        bar = st.progress(0, text="Warming up…")

        def progress_cb(done, total):
            bar.progress(done / total, text=f"Painting… pass {done} of {total}")

        with st.spinner("Creating your artwork…"):
            output_t = run_style_transfer(
                content_t, style_t, vgg, beta=beta, steps=steps,
                style_weights=style_weights, progress_cb=progress_cb)
            result = deprocess(output_t)

        bar.empty()

        # keep the result in session so downloading doesn't recompute
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.session_state["result_png"] = buf.getvalue()

        # be gentle with the 1 GB memory limit
        del content_t, style_t, output_t
        gc.collect()

    if "result_png" in st.session_state:
        step_badge("★", "Your artwork", "#2A9D8F")
        with st.container(border=True):
            st.image(st.session_state["result_png"], use_container_width=True)
        st.download_button(
            "Download PNG", data=st.session_state["result_png"],
            file_name="picasso.png", mime="image/png")

# ---------- ABOUT TAB --------------------------------------------------------
with about_tab:
    st.subheader("What's happening under the hood")
    st.markdown(
        "This tool uses a technique called **neural style transfer**. A neural "
        "network that was originally trained to recognise objects in photos turns "
        "out to also be very good at separating *what* is in a picture (its "
        "**content** — the shapes and subjects) from *how* it is painted (its "
        "**style** — the colours, brushwork, and texture).\n\n"
        "The app starts with a copy of your photo and gently repaints it, over and "
        "over, nudging it until its **content still matches your photo** but its "
        "**style matches the artwork** you chose. Every slider you see is a way of "
        "steering that balance:\n\n"
        "- **Style strength** decides who wins the tug-of-war between your photo and the artwork.\n"
        "- **Detail** sets how large the working image is — bigger looks sharper but takes longer.\n"
        "- **Refinement passes** are how many times the picture gets nudged toward the goal.\n"
        "- **Texture scale** chooses whether to borrow the artwork's fine grain or its bold shapes."
    )

    st.subheader("The idea, a little more precisely")
    st.markdown(
        "Under the hood, a pretrained **VGG19** network reads both images. The "
        "*content* of your photo is captured by the activations at one deep layer; "
        "the *style* of the artwork is captured by **Gram matrices** — how strongly "
        "different features fire together — across several layers. The generated "
        "image is then optimised (its pixels are the variables, not the network's "
        "weights) to minimise a combined loss:\n"
    )
    st.latex(r"\mathcal{L}_{total} = \alpha\,\mathcal{L}_{content} + \beta\,\mathcal{L}_{style}")
    st.markdown(
        "The **Style strength** slider is really the ratio of β to α, set on a "
        "logarithmic scale so the useful range fits into a friendly 1–10 dial."
    )

    st.subheader("What each control really changes")
    st.markdown(
        "The loss function itself has only **two hyperparameters, α and β** — but the "
        "full method has a few more knobs that steer the *optimisation* and the "
        "*inputs* rather than the loss. Here's how the sliders map to the maths:\n\n"
        "- **Style strength → β.** With α fixed at 1.0, this is the one true loss "
        "hyperparameter you're changing — the α/β balance. It's set on a log scale "
        "(strength 7 ≈ β of 1e7).\n"
        "- **Texture scale → the style-layer weights wₗ.** The style loss is "
        "Lstyle = Σ wₗ·Eₗ, a weighted sum over five layers. This dial tilts those "
        "weights toward shallow layers (fine grain) or deep layers (bold shapes). "
        "It's part of the loss, just not one of α/β.\n"
        "- **Refinement passes → number of optimisation steps.** *Not* a loss "
        "hyperparameter — it's how many iterations the optimiser runs. It changes how "
        "close you get to the optimum, not what the optimum is.\n"
        "- **Detail → the input image resolution.** Also not part of the loss — it's a "
        "preprocessing choice that affects sharpness and speed.\n\n"
        "So α, β and the wₗ weights define the *objective*; passes and resolution are "
        "practical settings every implementation has but that don't appear in the loss "
        "equation. That's why the Gatys paper and Prof. Khapra's lecture centre on "
        "α and β."
    )

    st.subheader("Credits & sources")
    st.markdown(
        "This project follows **Prof. Mitesh M. Khapra's** *Deep Art* lecture from "
        "the CS7015 (Deep Learning) course at IIT Madras — the walkthrough of the "
        "content loss, the Gram-matrix style representation, and the combined "
        "objective is taken directly from that lecture.\n\n"
        "The underlying method is from:\n\n"
        "- Leon A. Gatys, Alexander S. Ecker & Matthias Bethge, "
        "*A Neural Algorithm of Artistic Style* (2015), arXiv:1508.06576 — "
        "https://arxiv.org/abs/1508.06576\n"
        "- The expanded, peer-reviewed version: *Image Style Transfer Using "
        "Convolutional Neural Networks*, CVPR 2016.\n\n"
        "The recognition network is **VGG19** (Simonyan & Zisserman, 2014). "
        "Deep gratitude to these authors and educators; this app is a learning "
        "project built on their work."
    )