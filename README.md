# 🎨 Picasso, I Choose You!

**A neural style transfer studio for designers and artists.** Bring a photo and a
piece of art — a painting, a comic panel, a texture — and repaint your photo in that
artwork's style. Every result is an original reinterpretation, not a one-click filter.

🔗 **Live showcase & example gallery:** _add your Streamlit Cloud link here_

> _Tip: embed one of your best results here — `![example](assets/example.png)`_

---

## What it's for

Built for real creative work — no design suite or maths knowledge required:

- 🖼️ **Posters & prints** — turn a photo into striking wall art
- 📚 **Album & book covers** — a painterly, one-of-a-kind look
- 📱 **Social & marketing** — original visuals that don't look stock
- 🎨 **Mood boards & concepts** — explore a style direction fast
- 👕 **Merch & branding** — distinctive artwork from your own images

## The idea

A convolutional network trained to *recognise* images turns out to also separate
**what** is in a picture (its **content**) from **how** it is painted (its **style**):

- **Content** is read from the activations of a deep VGG19 layer (`conv4_2`) — deep
  enough to capture the subject while discarding exact pixels.
- **Style** is read from **Gram matrices** — how strongly features fire together —
  across several layers (`relu1_1 … relu5_1`). This keeps texture and colour while
  discarding *where* things are.

The clever part: **the network is frozen and never trained.** Instead, the *pixels*
of a generated image (started from your photo, or from random noise) are optimised so
that its content matches your photo and its style matches the artwork, by minimising:

```
L_total = w_c · L_content + w_s · L_style + w_tv · L_tv
```

Optimisation uses **L-BFGS with a strong-Wolfe line search**, run as one full
optimisation — the detail that lets the style fully set in.

This follows the method of **Gatys, Ecker & Bethge (2015)** and Prof. Mitesh Khapra's
*Deep Art* lecture (CS7015, IIT Madras).

## Tech stack

- **Python** · **PyTorch** — the deep learning engine
- **VGG19** (pretrained on ImageNet, via torchvision) — a *frozen* feature extractor
- **Gram matrices** — the style representation
- **L-BFGS with strong-Wolfe line search** — optimises the image pixels (not the network)
- **Total Variation loss** — keeps the output smooth rather than noisy
- **Streamlit** — the interactive interface
- **Pillow · NumPy** — image handling

## Controls

- **Starting point** — begin from the *content image* (sharp subject) or *random noise*
  (paints the whole frame, including flat backgrounds)
- **Style weight / Content weight** — the balance between artwork and photo
- **Smoothness (TV weight)** — raise to remove noise/speckle
- **Detail** — working resolution (higher = sharper but slower)
- **Refinement passes** — number of L-BFGS iterations
- **Texture scale** — emphasise fine grain vs bold shapes

## Run it locally

This app is compute-heavy (it runs PyTorch), so it's designed to run on your own
machine, where you have the memory and (optionally) a GPU.

```bash
git clone https://github.com/Ricky11234/Picasso_I_Choose_You.git
cd Picasso_I_Choose_You
pip install torch torchvision streamlit pillow numpy
streamlit run App.py
```

The app opens in your browser. Upload a photo and an artwork, choose a starting point,
and create. It uses your GPU automatically if you have one (much faster); otherwise it
runs on CPU. For a strong, painterly result, use a **bold painting** as the style, and
try the **random-noise** start if you want the background stylised too.

## Repo layout

```
Picasso_I_Choose_You/
├── App.py            # the full interactive app (run locally)
├── showcase.py       # lightweight gallery page (deployed to Streamlit Cloud)
├── requirements.txt  # deps for the showcase page (streamlit, pillow)
├── images/           # example images for the showcase
└── README.md
```

## Credits

- **Prof. Mitesh M. Khapra** — *Deep Art* lecture, CS7015 (Deep Learning), IIT Madras
- **Leon A. Gatys, Alexander S. Ecker & Matthias Bethge** — *A Neural Algorithm of
  Artistic Style* (2015), [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
- **VGG19** — Simonyan & Zisserman (2014)

## License

No license is set yet. To let others reuse it, add one — [MIT](https://choosealicense.com/licenses/mit/)
is a common, permissive choice.