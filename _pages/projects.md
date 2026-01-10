---
layout: single
title: "Projects"
permalink: /projects/
author_profile: true
classes: wide
---

Here are selected projects that best represent my technical depth (modeling + implementation + evaluation).

## Open-source

### GNN-Keras3 (TensorFlow/Keras)

Graph Neural Networks library with a Keras-style API for training/inference.

- Layers: graph convolution, graph attention, differentiable pooling, etc.
- Includes runnable examples (e.g., protein graph classification; ~80% single-fold validation accuracy on TUProteins in one demo).
- Repo: https://github.com/arijitcodespace/GNN-Keras3
- Docs: **[Get Started](/docs/overview/)**

## Research / academic projects

### Convergence of Asynchronous SGD for PL functions

Theory + experiments around asynchronous SGD under the Polyak–Łojasiewicz condition.

- Paper PDF: **[link](/assets/papers/paper.pdf)**
- Code: https://github.com/arijitcodespace/Asynchronous-SGD

### Text-to-Image Synthesis Optimization (UCLA MS project)

Optimized text-to-image GAN architectures under hardware constraints; integrated CLIP embeddings for stronger text conditioning.

- Achieved **FID 9.85** with a **10.6M** parameter generator (as a lightweight alternative to larger baselines).

### Mini-GPT (TinyStories)

Implemented a small GPT for story generation (course project), including parameter-saving tricks like weight tying.

- **Top-3 token accuracy:** ~88.6% on TinyStories.

### VOC signature clustering (B.E. thesis)

Machine-learning approach for clustering signatures of volatile organic compounds (VOCs) using E-Nose signals as a low-cost alternative to spectrometry.

- Achieved ~91% accuracy and ~0.89 F1 using a hierarchical pipeline (vs. weaker baselines).

---

Want a shorter view? The **Home** page has a quick list; this page is the “details” version.
