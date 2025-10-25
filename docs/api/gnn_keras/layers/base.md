---
title: "gnn_keras.layers.base"
permalink: /docs/api/gnn_keras/layers/base/
sidebar:
  nav: "docs"
---

## class `GraphLayer`

Base layer that consumes/produces `Graph` objects.

Subclasses should implement `_call_dense(x, a, training)` and
`_call_sparse(x, a, training)` and return transformed node features.

### Methods

- `call`
