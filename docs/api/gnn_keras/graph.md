---
title: "gnn_keras.graph"
permalink: /docs/api/gnn_keras/graph/
sidebar:
  nav: "docs"
---

## class `Graph`

Immutable graph container.

Attributes
-----------
x : tf.Tensor [N, F]
    Node features.
a : Union[tf.Tensor [N,N], tf.SparseTensor]
    Adjacency (dense or sparse). Self-loops optional.
y : Optional[tf.Tensor]
    Labels for nodes or graph.
mask : Optional[tf.Tensor]
    Boolean mask for nodes when doing semi-supervised tasks.

### Methods

- `is_sparse`


- `with_`


- `from_edge_index`
  
  Build a Graph from a COO-like edge_index using tf.sparse only.
