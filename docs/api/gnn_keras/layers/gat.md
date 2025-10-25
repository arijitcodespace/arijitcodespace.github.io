---
title: "gnn_keras.layers.gat"
permalink: /docs/api/gnn_keras/layers/gat/
sidebar:
  nav: "docs"
---

## class `GraphAttention`

GAT layer skeleton (Veličković et al.).

Efficient on sparse graphs via `tf.sparse`. For dense graphs, uses
masked attention over the adjacency.
