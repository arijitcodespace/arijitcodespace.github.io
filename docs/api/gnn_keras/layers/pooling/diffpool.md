---
title: "gnn_keras.layers.pooling.diffpool"
permalink: /docs/api/gnn_keras/layers/pooling/diffpool/
sidebar:
  nav: "docs"
---

## class `DiffPool`

Differentiable pooling layer skeleton.

Given node features X and adjacency A, learns an assignment S and produces
coarsened features X' = S^T X and adjacency A' = S^T A S.
This layer returns an updated Graph with coarsened (pooled) features/adj.

### Methods

- `call`


## class `BatchedDiffPool`

Segment-aware DiffPool that replicates K clusters per graph.

It reuses the Dense->K projection from a provided DiffPool instance (if any),
applies the activation (softmax by default) row-wise to get local S,
and constructs a block-diagonal assignment S_bd by **offsetting columns**
for each graph in the batch.

Vectorized build (no tf.map_fn), so variable graph sizes are fine.

### Methods

- `build_S_blockdiag`
  
  Z: [N_tot, F]

- `call`
  
  Z: [N_tot, F], A: [N_tot, N_tot] (dense or SparseTensor),
