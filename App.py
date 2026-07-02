"""
Picasso, I Choose You! — Neural Style Transfer as a design tool.

Engine faithfully follows the reference implementation
(github.com/nazianafis/Neural-Style-Transfer, based on Aleksa Gordic's), which
follows Gatys, Ecker & Bethge (2015) and Prof. Mitesh Khapra's Deep Art lecture.
The Streamlit interface wraps that engine for designers and artists.
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
# ENGINE  (faithful to the reference implementation)
# ----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference preprocessing: images in the 0-255 range, mean-subtracted, NO std
# division. This is the scale VGG's activations (and thus the reference's
# weights) are tuned to.
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


class Vgg19(torch.nn.Module):
    """
    VGG19 sliced so a single forward pass returns the exact feature maps the
    reference uses: five style layers (relu1_1..relu5_1) plus one content layer
    (conv4_2), in this order:
        [relu1_1, relu2_1, relu3_1, relu4_1, conv4_2, relu5_1]
    Content is index 4; style is every other index.
    """
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
        for x in range(2):        # relu1_1  (indices 0,1)
            self.slice1.add_module(str(x), feats[x])
        for x in range(2, 7):     # relu2_1  (..6)
            self.slice2.add_module(str(x), feats[x])
        for x in range(7, 12):    # relu3_1  (..11)
            self.slice3.add_module(str(x), feats[x])
        for x in range(12, 21):   # relu4_1  (..20)
            self.slice4.add_module(str(x), feats[x])
        for x in range(21, 22):   # conv4_2  (21)  <- content
            self.slice5.add_module(str(x), feats[x])
        for x in range(22, 30):   # relu5_1  (..29)
            self.slice6.add_module(str(x), feats[x])

        for p in self.parameters():
            p.requires_grad = False   # frozen feature extractor

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
    """PIL -> (1,3,H,W) tensor in the 0-255 mean-subtracted space, on DEVICE."""
    image = pil_image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(height),
        transforms.ToTensor(),                                  # [0,1]
        transforms.Lambda(lambda t: t.mul(255)),                # [0,255]
        transforms.Normalize(mean=IMAGENET_MEAN_255,
                             std=IMAGENET_STD_NEUTRAL),          # subtract mean
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def deprocess(tensor):
    """Reverse the 0-255 mean-subtracted space back to a viewable PIL image."""
    img = tensor.squeeze(0).detach().cpu().numpy()   # (3,H,W)
    img = np.moveaxis(img, 0, 2)                     # (H,W,3)
    img = img + np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    img = np.clip(img, 0, 255).astype("uint8")
    return Image.fromarray(img)


def gram_matrix(x, should_normalize=True):
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    """Smoothness term: sum of absolute neighbour differences."""
    return (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))


def compute_style_weights(tilt):
    """
    Per-style-layer weights. At tilt=0 all five are 0.2 (equal) — identical to the
    reference (which averages the layers). tilt<0 favours fine grain, tilt>0 bold.
    """
    idx = np.arange(5)                       # 0 shallow ... 4 deep
    raw = np.exp(tilt * (idx - 2.0))
    raw = raw / raw.sum()
    return [float(w) for w in raw]


def run_style_transfer(content_t, style_t, model, content_weight, style_weight,
                       tv_weight, steps, style_layer_weights, progress_cb=None):
    """Optimise the pixels of a generated image (faithful reference loop)."""
    content_feats = model(content_t)
    style_feats = model(style_t)
    target_content = content_feats[model.content_index].squeeze(0)
    target_style = [gram_matrix(style_feats[i]) for i in model.style_indices]

    generated = content_t.clone().requires_grad_(True)

    # The reference runs the WHOLE optimisation in ONE step with max_iter=steps
    # and a strong-Wolfe line search. This is what actually makes the style set
    # in — a loop of max_iter=1 steps does not build the L-BFGS trajectory.
    optimizer = optim.LBFGS([generated], max_iter=steps,
                            line_search_fn="strong_wolfe")

    mse_mean = torch.nn.MSELoss(reduction="mean")
    mse_sum = torch.nn.MSELoss(reduction="sum")
    cnt = [0]

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        current = model(generated)
        current_content = current[model.content_index].squeeze(0)
        c_loss = mse_mean(target_content, current_content)

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
            progress_cb(min(cnt[0], steps), steps)
        return total

    optimizer.step(closure)
    return generated.detach()


# ----------------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Picasso, I Choose You!",
                   page_icon="🎨", layout="wide")

st.markdown(
    """
    <style>
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


def chip(text, color):
    st.markdown(
        f'<div style="display:inline-block;background:{color}22;color:{color};'
        f'border:1px solid {color}66;padding:0.3rem 0.75rem;border-radius:999px;'
        f'font-size:0.86rem;font-weight:600;margin:0.15rem 0 0.2rem;">{text}</div>',
        unsafe_allow_html=True,
    )


st.title("🎨 Picasso, I Choose You!")
st.markdown(
    "**Repaint your photo in the style of any artwork.** Bring a photo and a piece "
    "of art — a painting, a comic panel, a texture — and blend them into something "
    "new for your posters, covers, and prints."
)

create_tab, about_tab = st.tabs(["Create", "How it works"])

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
        content_file = st.file_uploader("Your photo (the subject to keep)",
                                        type=["jpg", "jpeg", "png"], key="content")
        if content_file:
            st.image(Image.open(content_file), caption="Photo", use_container_width=True)
    with col_style:
        style_file = st.file_uploader("The artwork (the look to borrow)",
                                      type=["jpg", "jpeg", "png"], key="style")
        if style_file:
            st.image(Image.open(style_file), caption="Style", use_container_width=True)

    step_badge(2, "Dial in the look", "#2A9D8F")

    # Style strength -> style_weight (content_weight fixed at 1e5, as in the reference).
    strength = st.slider("Style strength", 1, 10, 7,
                         help="How strongly the artwork's look takes over your photo.")
    if strength <= 3:
        chip("Gentle — your photo stays clearly recognisable", "#2A9D8F")
    elif strength <= 7:
        chip("Balanced — a clear repaint that still reads as your photo", "#2E4BC6")
    else:
        chip("Bold — the artwork's colours and textures take over", "#E63946")
    content_weight = 1e5
    style_weight = 10.0 ** (3.0 + (strength - 1) / 3.0)   # s7 -> 1e5, s10 -> 1e6
    tv_weight = 1.0

    col_a, col_b = st.columns(2)
    with col_a:
        size = st.slider("Detail", 256, 512, 400, step=32,
                         help="Bigger = sharper, more detailed — but slower.")
        st.caption("On the free CPU cloud, keep this around 400 or below.")
    with col_b:
        steps = st.slider("Refinement passes", 50, 1000, 300, step=25,
                          help="How many L-BFGS iterations. More = stronger, slower.")
        st.caption("300 gives a strong result; the reference uses ~1000.")

    with st.expander("Advanced — texture scale"):
        tilt = st.slider("Fine  ←→  Bold", -1.5, 1.5, 0.0, step=0.1,
                         help="Which scale of the artwork's texture to emphasise.")
        if tilt < -0.3:
            chip("Fine detail — delicate, brush-like grain", "#7B2CBF")
        elif tilt > 0.3:
            chip("Bold shapes — large sweeps and blocks of colour", "#F4A261")
        else:
            chip("Balanced — fine detail and large shapes (reference default)", "#2A9D8F")
    style_layer_weights = compute_style_weights(tilt)

    step_badge(3, "Create", "#7B2CBF")
    st.caption("This runs on CPU here, so expect a few minutes. Lower Detail and "
               "passes for a quicker preview.")

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
                content_t, style_t, vgg,
                content_weight=content_weight, style_weight=style_weight,
                tv_weight=tv_weight, steps=steps,
                style_layer_weights=style_layer_weights, progress_cb=progress_cb)
            result = deprocess(output_t)

        bar.empty()
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.session_state["result_png"] = buf.getvalue()

        del content_t, style_t, output_t
        gc.collect()

    if "result_png" in st.session_state:
        step_badge("★", "Your artwork", "#2A9D8F")
        with st.container(border=True):
            st.image(st.session_state["result_png"], use_container_width=True)
        st.download_button("Download PNG", data=st.session_state["result_png"],
                           file_name="picasso.png", mime="image/png")

with about_tab:
    st.subheader("What's happening under the hood")
    st.markdown(
        "This tool uses **neural style transfer**. A frozen, pretrained **VGG19** "
        "network reads both images. The **content** of your photo is taken from a "
        "deep layer (conv4_2); the **style** of the artwork is taken from **Gram "
        "matrices** — how strongly features fire together — across five layers "
        "(relu1_1 … relu5_1). Starting from a copy of your photo, the app optimises "
        "the *pixels* (not the network's weights) to match your photo's content and "
        "the artwork's style."
    )
    st.latex(r"\mathcal{L}_{total} = w_c\,\mathcal{L}_{content} + w_s\,\mathcal{L}_{style} + w_{tv}\,\mathcal{L}_{tv}")
    st.markdown(
        "Optimisation uses **L-BFGS with a strong-Wolfe line search**, run as a single "
        "full optimisation — the key detail that lets the style fully set in."
    )

    st.subheader("What each control changes")
    st.markdown(
        "- **Style strength** → the style weight `w_s` (content weight is fixed), on a "
        "log scale. Higher = the artwork dominates.\n"
        "- **Texture scale** → the per-layer style weights (fine grain vs bold shapes); "
        "the centre is the reference's equal weighting.\n"
        "- **Refinement passes** → number of L-BFGS iterations (not part of the loss).\n"
        "- **Detail** → the working image resolution (not part of the loss)."
    )

    st.subheader("Credits & sources")
    st.markdown(
        "Engine follows the reference implementation "
        "(github.com/nazianafis/Neural-Style-Transfer, based on Aleksa Gordic's).\n\n"
        "- Prof. Mitesh M. Khapra — *Deep Art* lecture, CS7015, IIT Madras\n"
        "- Gatys, Ecker & Bethge — *A Neural Algorithm of Artistic Style* (2015), "
        "arXiv:1508.06576\n"
        "- VGG19 — Simonyan & Zisserman (2014)"
    )