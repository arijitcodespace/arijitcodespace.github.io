---
layout: single
title: "Research"
permalink: /research/
author_profile: true
classes: wide
---

My work sits at the intersection of **deep learning research** and **practical ML systems**. I'm especially interested in how we can make training more reliable (optimization theory), models more useful (multimodal + graph learning), and evaluation more honest.

## Current focus

### Vision-language for medical imaging

At Symviq, I work on **vision-language modeling** for chronic disease classification using retinal fundus images. The direction I find most exciting is using CLIP-style contrastive objectives so that clinical, human-readable descriptions can act as supervision and improve generalization.

**Typical themes:**
- CLIP/SimCLR-style encoders for medical imaging
- Text-conditioning / multimodal embeddings for downstream classification and retrieval
- Foundation-model style pretraining for multiple imaging tasks

### Graph representation learning (GNNs)

I’m building **GNN-Keras3**, a Keras-first library that makes it easy to prototype GNN layers (GCN, GAT, differentiable pooling, etc.) with clean APIs and runnable examples.

See: **[Projects](/projects/)** and **[GNN docs](/docs/overview/)**.

## Theory + optimization

My recent research paper studies **asynchronous stochastic gradient descent** under the **Polyak–Łojasiewicz (PL)** condition, which is a useful middle ground between convexity and general nonconvex objectives.

See: **[Publications](/publications/)**.

## If you want to collaborate

If you’re working on multimodal learning, graph ML, or optimization theory for deep learning, I’m open to collaborations. The best way to reach me is via **email** (see the sidebar).
