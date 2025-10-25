---
title: "gnn_keras.datasets.base"
permalink: /docs/api/gnn_keras/datasets/base/
sidebar:
  nav: "docs"
---

## class `GraphDataset`

Minimal dataset protocol returning Graph objects.

Implement `__iter__` for graph-level tasks or provide a single Graph via
`.graph()` for transductive node tasks.

### Methods

- `graph`
