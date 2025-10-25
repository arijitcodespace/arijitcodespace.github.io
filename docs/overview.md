---
title: "Get Started"
permalink: /docs/overview/
sidebar:
  nav: "docs"
---

Welcome! This site hosts my work and the documentation for **gnn-keras3**.

## Quickstart

**From source (recommended for now):**
```bash
# clone the repo and install in editable mode
git clone https://github.com/arijitcodespace/gnn-keras3.git
cd gnn-keras3
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -U pip wheel
pip install -e .
```

Then try an example:
```bash
python examples/01_gcn_node_classification.py
```

## Where to next?
- **Installation details:** [/docs/installation/](/docs/installation/)
- **Examples:** [/docs/examples/](/docs/examples/)
- **API Reference:** [/docs/api/](/docs/api/)
