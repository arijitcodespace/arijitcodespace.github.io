---
title: "gnn_keras.utils.scipy"
permalink: /docs/api/gnn_keras/utils/scipy/
sidebar:
  nav: "docs"
---

## function `coo_to_tf_sparse`

Convert a SciPy COO matrix to a `tf.SparseTensor` (indices int64).

## function `csr_to_tf_sparse`

Convert a SciPy CSR matrix to a `tf.SparseTensor` via COO view.

## function `any_scipy_to_tf_sparse`

Convert any SciPy sparse (CSR/CSC/COO/...) to `tf.SparseTensor`.

## function `scipy_edge_index`

Return edge_index [2, E] (int32) from a SciPy sparse matrix.
