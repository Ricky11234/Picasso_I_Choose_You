"""
Picasso, I Choose You! — Neural Style Transfer as a design tool.

Engine follows the neural style transfer method of Gatys, Ecker & Bethge (2015),
following Prof. Mitesh Khapra's Deep Art lecture (CS7015, IIT Madras).

Flow: a first-time user just brings two images and chooses a starting point
(content image or random noise), and the app renders with recommended default
hyperparameters. After that, the full controls unlock for fine-tuning.
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
# ENGINE
# ----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0-255 range, mean-subtracted, no std division.
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

# Recommended default hyperparameters.
DEFAULTS = dict(
    content_weight=1e5, style_weight=3e4, tv_weight=1.0,
    height=400, iterations=1000,
)


class Vgg19(torch.nn.Module):
    """Sliced VGG19 returning [relu1_1, relu2_1, relu3_1, relu4_1, conv4_2, relu5_1]."""
    def __init__(self):
        super().__init__()
        feats = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.content_index = 4
        self.style_indices = [0, 1, 2, 3, 5]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), feats[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), feats[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), feats[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), feats[x])
        for x in range(21, 22):
            self.slice5.add_module(str(x), feats[x])
        for x in range(22, 30):
            self.slice6.add_module(str(x), feats[x])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.slice1(x); relu1_1 = x
        x = self.slice2(x); relu2_1 = x
        x = self.slice3(x); relu3_1 = x
        x = self.slice4(x); relu4_1 = x
        x = self.slice5(x); conv4_2 = x
        x = self.slice6(x); relu5_1 = x
        return [relu1_1, relu2_1, relu3_1, relu4_1, conv4_2, relu5_1]


@st.cache_resource(show_spinner=False)
def load_vgg():
    return Vgg19().to(DEVICE).eval()


def preprocess(pil_image, height):
    image = pil_image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(height),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def deprocess(tensor):
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.moveaxis(img, 0, 2)
    img = img + np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    img = np.clip(img, 0, 255).astype("uint8")
    return Image.fromarray(img)


def gram_matrix(x, should_normalize=True):
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    gram = features.bmm(features.transpose(1, 2))
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))


def compute_style_weights(tilt):
    idx = np.arange(5)
    raw = np.exp(tilt * (idx - 2.0))
    raw = raw / raw.sum()
    return [float(w) for w in raw]


def run_style_transfer(content_t, style_t, model, content_weight, style_weight,
                       tv_weight, iterations, style_layer_weights,
                       init_method="content", progress_cb=None):
    content_feats = model(content_t)
    style_feats = model(style_t)
    target_content = content_feats[model.content_index].squeeze(0)
    target_style = [gram_matrix(style_feats[i]) for i in model.style_indices]

    # Initialisation: content image (faithful, sharp subject) OR random noise
    # (paints the whole frame, including flat backgrounds).
    if init_method == "noise":
        start = torch.randn(content_t.shape, device=DEVICE) * 90.0
        generated = start.detach().requires_grad_(True)
    else:
        generated = content_t.clone().requires_grad_(True)

    optimizer = optim.LBFGS([generated], max_iter=iterations,
                            line_search_fn="strong_wolfe")
    mse_mean = torch.nn.MSELoss(reduction="mean")
    mse_sum = torch.nn.MSELoss(reduction="sum")
    cnt = [0]

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        current = model(generated)
        c_loss = mse_mean(target_content, current[model.content_index].squeeze(0))
        s_loss = 0.0
        current_style = [gram_matrix(current[i]) for i in model.style_indices]
        for w, gt, hat in zip(style_layer_weights, target_style, current_style):
            s_loss = s_loss + w * mse_sum(gt[0], hat[0])
        tv = total_variation(generated)
        total = content_weight * c_loss + style_weight * s_loss + tv_weight * tv
        if total.requires_grad:
            total.backward()
        cnt[0] += 1
        if progress_cb is not None:
            progress_cb(min(cnt[0], iterations), iterations)
        return total

    optimizer.step(closure)
    return generated.detach()


# ----------------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Picasso, I Choose You!", page_icon="🎨", layout="wide")

st.markdown(
    """
    <style>
    .stButton > button, .stDownloadButton > button {
      background: linear-gradient(90deg,#E63946,#F4A261,#2A9D8F,#2E4BC6);
      background-size: 250% 100%; color:#fff; border:none; border-radius:12px;
      padding:0.6rem 1.4rem; font-weight:700; font-size:1.02rem;
      transition: background-position .5s ease, transform .1s ease, box-shadow .2s ease;
      box-shadow: 0 3px 10px rgba(46,75,198,0.18);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      background-position:100% 0; transform:translateY(-1px); color:#fff;
      box-shadow: 0 6px 16px rgba(230,57,70,0.25);
    }
    .stButton > button:disabled { background:#d9d4c9; color:#8f8a7c; box-shadow:none; }
    [data-testid="stFileUploaderDropzone"] {
      border:2px dashed #2E4BC6; background: rgba(46,75,198,0.05); border-radius:14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def step_badge(num, text, color):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.9rem 0 0.4rem;">'
        f'<span style="background:{color};color:#fff;width:1.9rem;height:1.9rem;'
        f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
        f'font-weight:800;flex:0 0 auto;">{num}</span>'
        f'<span style="font-size:1.3rem;font-weight:800;color:#1F2933;">{text}</span></div>',
        unsafe_allow_html=True,
    )


def _num(label):
    return float(label.replace(",", ""))


st.title("🎨 Picasso, I Choose You!")
st.markdown(
    "**Repaint your photo in the style of any artwork.** Bring a photo and a piece "
    "of art, choose where to start, and blend them into something new."
)

create_tab, about_tab = st.tabs(["Create", "How it works"])

with create_tab:
    st.markdown(
        '<div style="background:linear-gradient(120deg,#E63946,#F4A261,#E9C46A,'
        '#2A9D8F,#2E4BC6,#7B2CBF);padding:1.3rem 1.6rem;border-radius:18px;'
        'margin:0.3rem 0 0.6rem;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,0.12);">'
        '<div style="font-size:1.55rem;font-weight:800;">Turn a photo into a painting</div>'
        '<div style="opacity:0.95;margin-top:0.25rem;font-size:1.02rem;">'
        'Upload a photo and an artwork, pick a starting point, and let the studio blend them.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    advanced = st.session_state.get("has_generated", False)

    # --- Step 1: images ---
    step_badge(1, "Bring your two images", "#E63946")
    col_content, col_style = st.columns(2)
    with col_content:
        content_file = st.file_uploader("Your photo (the subject to keep)",
                                        type=["jpg", "jpeg", "png"], key="content")
        if content_file:
            content_file.seek(0)
            st.image(Image.open(content_file), caption="Photo", use_container_width=True)
    with col_style:
        style_file = st.file_uploader("The artwork (the look to borrow)",
                                      type=["jpg", "jpeg", "png"], key="style")
        if style_file:
            style_file.seek(0)
            st.image(Image.open(style_file), caption="Style", use_container_width=True)

    # --- Step 2: starting point (the init choice) ---
    step_badge(2, "Choose a starting point", "#2A9D8F")
    init_label = st.radio(
        "Where should the painting begin?",
        ["Content image — keeps your subject sharp (recommended)",
         "Random noise — paints the whole frame, background included"],
        index=0,
    )
    init_method = "noise" if init_label.startswith("Random") else "content"
    if init_method == "noise":
        st.caption("Starting from noise stylises the entire frame — good when a flat "
                   "background isn't picking up the style — but the subject is a little "
                   "less faithful and each run varies.")
    else:
        st.caption("Starting from your photo keeps the subject crisp. Flat backgrounds "
                   "may stay plain unless the style is a bold painting.")

    # --- Step 3: hyperparameters (only after the first render) ---
    if advanced:
        step_badge(3, "Fine-tune (optional)", "#7B2CBF")
        with st.expander("Hyperparameters", expanded=True):
            style_weight = _num(st.select_slider(
                "Style weight",
                options=["10", "100", "1,000", "3,000", "10,000", "30,000",
                         "100,000", "300,000", "1,000,000"],
                value="30,000",
                help="Higher = more style. Recommended default is 30,000."))
            content_weight = _num(st.select_slider(
                "Content weight",
                options=["1,000", "10,000", "100,000", "1,000,000"],
                value="100,000",
                help="How strongly to preserve your photo. Recommended default is 100,000."))
            tv_weight = _num(st.select_slider(
                "Smoothness (TV weight)",
                options=["0", "1", "10", "100", "1,000", "10,000", "100,000", "1,000,000"],
                value="1",
                help="Raise this to reduce noise/speckle. Recommended default is 1."))
            c1, c2 = st.columns(2)
            with c1:
                height = st.slider("Detail (image size)", 256, 512, 400, step=32,
                                   help="Larger = sharper but slower. Recommended 400.")
            with c2:
                iterations = st.slider("Refinement passes", 50, 1000, 1000, step=50,
                                       help="L-BFGS iterations. Recommended 1000.")
                st.caption("1000 is the recommended default — slow on CPU (several "
                           "minutes). Lower to ~300 for a quick preview.")
            tilt = st.slider("Texture scale  (Fine ←→ Bold)", -1.5, 1.5, 0.0, step=0.1,
                             help="Emphasise fine grain vs bold shapes. 0 = equal weighting.")
    else:
        style_weight = DEFAULTS["style_weight"]
        content_weight = DEFAULTS["content_weight"]
        tv_weight = DEFAULTS["tv_weight"]
        height = DEFAULTS["height"]
        iterations = DEFAULTS["iterations"]
        tilt = 0.0
        st.caption("Your first render uses recommended default settings. You'll be able "
                   "to fine-tune every hyperparameter afterwards.")

    style_layer_weights = compute_style_weights(tilt)

    # --- Create ---
    ready = content_file is not None and style_file is not None
    if not ready:
        st.info("Add both a photo and an artwork above to begin.")

    if st.button("Create artwork", type="primary", disabled=not ready):
        vgg = load_vgg()
        content_file.seek(0)
        style_file.seek(0)
        content_t = preprocess(Image.open(content_file), height)
        style_t = preprocess(Image.open(style_file), height)

        bar = st.progress(0, text="Warming up…")

        def progress_cb(done, total):
            bar.progress(done / total, text=f"Painting… pass {done} of {total}")

        with st.spinner("Creating your artwork…"):
            output_t = run_style_transfer(
                content_t, style_t, vgg,
                content_weight=content_weight, style_weight=style_weight,
                tv_weight=tv_weight, iterations=iterations,
                style_layer_weights=style_layer_weights,
                init_method=init_method, progress_cb=progress_cb)
            result = deprocess(output_t)

        bar.empty()
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.session_state["result_png"] = buf.getvalue()
        st.session_state["has_generated"] = True

        del content_t, style_t, output_t
        gc.collect()
        st.rerun()

    if "result_png" in st.session_state:
        step_badge("★", "Your artwork", "#2A9D8F")
        with st.container(border=True):
            st.image(st.session_state["result_png"], use_container_width=True)
        st.download_button("Download PNG", data=st.session_state["result_png"],
                           file_name="picasso.png", mime="image/png")
        st.caption("Not bold enough? Try a bolder painting as the style, raise the style "
                   "weight, or start from noise. Too noisy? Raise the smoothness (TV) weight.")

with about_tab:
    st.subheader("What's happening under the hood")
    st.markdown(
        "This tool uses **neural style transfer**. A frozen, pretrained **VGG19** reads "
        "both images. The **content** of your photo comes from a deep layer (conv4_2); "
        "the **style** of the artwork comes from **Gram matrices** — how strongly "
        "features fire together — across five layers (relu1_1 … relu5_1). Starting from "
        "either your photo or random noise, the app optimises the *pixels* (not the "
        "network's weights) to match your photo's content and the artwork's style."
    )
    st.latex(r"\mathcal{L}_{total} = w_c\,\mathcal{L}_{content} + w_s\,\mathcal{L}_{style} + w_{tv}\,\mathcal{L}_{tv}")
    st.markdown(
        "Optimisation uses **L-BFGS with a strong-Wolfe line search**, run as one full "
        "optimisation — the detail that lets the style fully set in."
    )

    st.subheader("What each control changes")
    st.markdown(
        "- **Starting point** → content image (sharp subject, flat backgrounds resist) "
        "or random noise (whole frame painted). This is the *initialisation method*.\n"
        "- **Style weight `w_s`** → how strongly the artwork dominates (default 30,000).\n"
        "- **Content weight `w_c`** → how strongly your photo is preserved (default 100,000).\n"
        "- **Smoothness `w_tv`** → raise to remove noise/speckle (default 1).\n"
        "- **Detail** → working resolution (not part of the loss).\n"
        "- **Refinement passes** → number of L-BFGS iterations (not part of the loss).\n"
        "- **Texture scale** → per-layer style weighting (0 = equal weighting)."
    )

    st.subheader("Credits & sources")
    st.markdown(
        "- Prof. Mitesh M. Khapra — *Deep Art* lecture, CS7015 (Deep Learning), IIT Madras\n"
        "- Leon A. Gatys, Alexander S. Ecker & Matthias Bethge — *A Neural Algorithm of "
        "Artistic Style* (2015), arXiv:1508.06576\n"
        "- VGG19 — Simonyan & Zisserman (2014)"
    )