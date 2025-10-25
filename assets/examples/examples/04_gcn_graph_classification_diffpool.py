# 04_gcn_graph_classification_diffpool.py
# ----------------------------------------------------------------------------
# Battle‑test gnn‑keras3 on TU‑Dataset PROTEINS / PROTEINS_full with
# **mini‑batch training & evaluation** (multiple graphs per step).
#
# Usage
#   python 04_gcn_graph_classification_diffpool.py --name PROTEINS_full --pool mean --hidden 64 --epochs 150 --batch_size 256 --lr 5e-4 --dropout 0.2 --alpha_lp 0.1 --beta_ent 0.001 --diffpool_mode batched --logfile ./logs/logs.tx
# ----------------------------------------------------------------------------


from __future__ import annotations

import argparse, random, math, os, sys
from dataclasses import dataclass
from typing import List, Tuple

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers

from tqdm import trange

from gnn_keras import Graph
from gnn_keras.datasets import TUGraph, TUDataset, TUGraph_to_Graph
from gnn_keras.layers import GraphConv
from gnn_keras.layers.pooling import DiffPool, BatchedDiffPool
from gnn_keras.utils.batching import pack_block_diagonal

# ---------------------
# Utility: seed control
# ---------------------
def fix_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ------------------------------------
# Simple readout heads for graph-level
# ------------------------------------
class GlobalSumPool(layers.Layer):
    def call(self, x: tf.Tensor, segment_ids: tf.Tensor, num_segments: int):
        # x: [N, F], segment_ids [N], returns [num_segments, F]
        return tf.math.unsorted_segment_sum(x, segment_ids, num_segments)

class GlobalMeanPool(layers.Layer):
    def call(self, x: tf.Tensor, segment_ids: tf.Tensor, num_segments: int):
        sums = tf.math.unsorted_segment_sum(x, segment_ids, num_segments)
        counts = tf.math.unsorted_segment_sum(tf.ones_like(segment_ids, dtype = x.dtype),
                                              segment_ids, num_segments)
        return sums / tf.maximum(counts[:, None], 1.0)


# ----------------
# Model definition
# ----------------
@dataclass
class Config:
    hidden: int = 64
    alpha_lp: float = 1.0
    beta_ent: float = 0.1
    clusters: int = 25
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 150
    batch_size: int = 16
    pool: str = "sum"  # or "mean"
    diffpool_mode: str = "batched"  # or "unbatched"
    logfile: str = None

class GCNDiffPool(keras.Model):
    def __init__(self, num_classes: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        self.enc1 = GraphConv(H, normalize = True, self_loops = True)
        self.enc2 = GraphConv(H, normalize = True, self_loops = True)
        self.dropout = layers.Dropout(cfg.dropout)
        self.diffpool = DiffPool(cfg.clusters)
        self.batched_diffpool = BatchedDiffPool(cfg.clusters, diffpool = self.diffpool)
        self.post = GraphConv(H, normalize = True, self_loops = True)
        self.readout = GlobalSumPool() if cfg.pool == "sum" else GlobalMeanPool()
        self.cls = layers.Dense(num_classes)  # logits

    def encode(self, G: Graph, training: bool):
        # Two GraphConv layers on (possibly block‑diag) graph
        Z = self.enc1(G, training = training).x
        Z = tf.nn.relu(Z)
        Z = self.dropout(Z, training = training)
        Z = self.enc2(G.with_(x = Z), training = training).x
        Z = tf.nn.relu(Z)
        Z = self.dropout(Z, training = training)
        return Z

    def call_unbatched(self, graphs: List[Graph], training: bool = False):
        # Per‑graph DiffPool, then pack, then post‑GCN & readout
        pooled_graphs = []
        for g in graphs:
            Z = self.encode(g, training)
            gZ = g.with_(x = Z)
            gp = self.diffpool(gZ, training = training)  # Graph with pooled x,a
            gp = self.post(gp, training = training)      # one more GraphConv
            pooled_graphs.append(gp)
        # Pack pooled graphs into a block‑diag big graph for readout together
        Gp = pack_block_diagonal(pooled_graphs)
        # segment ids for pooled graph (each graph contributes 'clusters' nodes)
        B = len(graphs)
        seg_ids = tf.repeat(tf.range(B, dtype = tf.int32), repeats = self.cfg.clusters)
        H = tf.nn.relu(Gp.x)
        H = self.dropout(H, training = training)
        G_emb = self.readout(H, seg_ids, B)           # [B, H]
        logits = self.cls(G_emb)                      # [B, C]
        return logits

    def call_batched(self, batch_graph: Graph, n_nodes: tf.Tensor, training: bool = False):
        # Encode on block‑diag graph, do batched DiffPool, then post‑GCN & readout
        Z = self.encode(batch_graph, training)            # [N_tot, H]
        Zp, Ap, seg_ids_p = self.batched_diffpool(Z, batch_graph.a, n_nodes)
        Gp = batch_graph.with_(x = Zp, a = Ap)
        Gp = self.post(Gp, training = training)
        H = tf.nn.relu(Gp.x)
        H = self.dropout(H, training = training)
        B = tf.shape(n_nodes)[0]
        G_emb = self.readout(H, seg_ids_p, B)            # [B, H]
        logits = self.cls(G_emb)                          # [B, C]
        return logits

    def call(self, graphs_or_batch, n_nodes = None, training: bool = False):
        if isinstance(graphs_or_batch, list):
            return self.call_unbatched(graphs_or_batch, training = training)
        else:
            if n_nodes is None:
                raise ValueError(f"`n_nodes` must be passed if using batched data. "
                                 f"Received n_nodes = {n_nodes}")
            return self.call_batched(graphs_or_batch, n_nodes, training = training)

# ---------------------
# Data / batching utils
# ---------------------
def make_batches(graphs: List[Graph], labels: np.ndarray, batch_size: int):
    """Yield mini‑batches as (list_of_graphs, labels)"""
    N = len(graphs)
    idx = np.arange(N)
    for i in range(0, N, batch_size):
        j = idx[i:i+batch_size]
        gs = [graphs[k] for k in j]
        ys = labels[j]
        yield gs, ys

def pack_with_counts(graphs: List[Graph]) -> Tuple[Graph, tf.Tensor]:
    """Pack graphs to a block‑diag big Graph and return node counts per graph."""
    Gbd = pack_block_diagonal(graphs)
    n_nodes = tf.convert_to_tensor([tf.shape(g.x)[0] for g in graphs], dtype = tf.int32)
    return Gbd, n_nodes

# -----------------------------------------------------------
# Keras-friendly batching + Trainer with DiffPool auxiliaries
# -----------------------------------------------------------
class GraphBatchSequence(keras.utils.Sequence):
    """Keras Sequence yielding (graphs_list, labels) per step."""
    def __init__(self, graphs, labels, batch_size: int):
        self.graphs = list(graphs)
        self.labels = np.asarray(labels, dtype = np.int32)
        self.batch_size = int(batch_size)

    def __len__(self):
        return int(math.ceil(len(self.graphs) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        j = min((idx + 1) * self.batch_size, len(self.graphs))
        gs = self.graphs[i:j]
        ys = self.labels[i:j]
        return gs, ys

class Trainer(keras.Model):
    """Wraps GCNDiffPool and adds DiffPool link-prediction + entropy losses."""
    def __init__(self, backbone: GCNDiffPool, cfg: Config):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        # Track aux losses as metrics
        self.base_loss_tracker = keras.metrics.Mean(name = 'base_loss')
        self.loss_tracker = keras.metrics.Mean(name = 'loss')
        self.lp_metric = keras.metrics.Mean(name = "lp")
        self.ent_metric = keras.metrics.Mean(name = "ent")
    
    @staticmethod
    def _create_logfile(filepath):
        logfile = Path(filepath)
        logfile.parent.mkdir(parents = True, exist_ok = True)
        logfile.touch(exist_ok = True)
        return logfile

    @property
    def metrics(self):
        # include custom metrics with compiled ones
        return super().metrics + [self.lp_metric, self.ent_metric]

    def _entropy_loss(self, S: tf.Tensor) -> tf.Tensor:
        eps = tf.constant(1e-8, dtype = S.dtype)
        ent = -tf.reduce_sum(S * tf.math.log(S + eps), axis = -1)  # [N]
        return tf.reduce_mean(ent)

    def _linkpred_loss_dense(self, A_dense: tf.Tensor, S: tf.Tensor, n_nodes: tf.Tensor | None = None) -> tf.Tensor:
        """
        Frobenius norm of (A - S S^T) normalized by sum_b n_b^2 (matches DiffPool).
        For unbatched (single graph), falls back to N^2.
        """
        SSt = tf.linalg.matmul(S, S, transpose_b = True)
        diff2 = tf.square(A_dense - SSt)                  # elementwise
        fro = tf.sqrt(tf.reduce_sum(diff2))              # ||.||_F
        if n_nodes is None:
            N = tf.cast(tf.shape(A_dense)[0], tf.float32)
            denom = tf.square(N)
        else:
            denom = tf.reduce_sum(tf.square(tf.cast(n_nodes, tf.float32)))
        return fro / (denom + 1e-8)

    def _forward_batched(self, graphs_batch, training: bool):
        # Build block-diagonal batch
        Gbd, n_nodes = pack_with_counts(graphs_batch)
        # Encode once
        Z = self.backbone.encode(Gbd, training=training)             # [N_tot, H]
        # Build block-diag S and local S (no column offsets) using shared head
        Zp, Ap, seg_ids, S_local, Sbd = self.backbone.batched_diffpool(Z, Gbd.a, n_nodes)

        # Aux losses on block-diagonal adjacency in dense form
        if isinstance(Gbd.a, tf.SparseTensor):
            A_dense = tf.sparse.to_dense(Gbd.a)
        else:
            A_dense = Gbd.a
        lp = self._linkpred_loss_dense(A_dense, Sbd, n_nodes = n_nodes)      # uses offset columns
        ent = self._entropy_loss(S_local)                                    # per-node across all graphs

        # Classifier path (post GraphConv + readout)
        Gp = Gbd.with_(x = Zp, a = Ap)
        Gp = self.backbone.post(Gp, training = training)
        H = tf.nn.relu(Gp.x)
        H = self.backbone.dropout(H, training = training)
        B = tf.shape(n_nodes)[0]
        G_emb = self.backbone.readout(H, seg_ids, B)
        logits = self.backbone.cls(G_emb)
        return logits, lp, ent

    def _forward_unbatched(self, graphs_batch, training: bool):
        pooled_graphs = []
        lp_list = []
        ent_list = []
        K = self.backbone.cfg.clusters
        for g in graphs_batch:
            # Encode
            Z = self.backbone.encode(g, training = training)     # [N, H]
            # Assignment via shared head
            logits_S = self.backbone.diffpool.proj(Z)
            S = self.backbone.diffpool.act(logits_S)           # [N, K]
            # Aux losses per graph
            A_dense = tf.sparse.to_dense(g.a) if isinstance(g.a, tf.SparseTensor) else g.a
            lp_list.append(self._linkpred_loss_dense(A_dense, S, n_nodes = None))
            ent_list.append(self._entropy_loss(S))
            # Pool
            Zp = tf.linalg.matmul(S, Z, transpose_a = True)      # [K, H]
            Ap = tf.linalg.matmul(tf.linalg.matmul(S, A_dense, transpose_a = True), S)  # [K, K]
            gp = g.with_(x = Zp, a = Ap)
            gp = self.backbone.post(gp, training = training)
            pooled_graphs.append(gp)
        # Pack pooled and classify
        Gp = pack_block_diagonal(pooled_graphs)
        B = len(graphs_batch)
        seg_ids = tf.repeat(tf.range(B, dtype = tf.int32), repeats = K)
        H = tf.nn.relu(Gp.x)
        H = self.backbone.dropout(H, training = training)
        G_emb = self.backbone.readout(H, seg_ids, B)
        logits = self.backbone.cls(G_emb)
        lp = tf.reduce_mean(tf.stack(lp_list)) if lp_list else tf.constant(0.0, dtype = H.dtype)
        ent = tf.reduce_mean(tf.stack(ent_list)) if ent_list else tf.constant(0.0, dtype = H.dtype)
        return logits, lp, ent

    def call(self, inputs, training = False):
        # Not used directly by fit(); we drive through _forward_* for aux losses.
        if isinstance(inputs, tuple):
            Gbd, n_nodes = inputs
            return self.backbone(Gbd, n_nodes, training = training)
        else:
            return self.backbone(inputs, training = training)

    def compile(self, optimizer = None, loss = None, metrics = None, **kwargs):
        opt = optimizer or optimizers.Adam(self.cfg.lr)
        self.loss_obj = loss or losses.SparseCategoricalCrossentropy(from_logits = True)
        mets = metrics or [keras.metrics.SparseCategoricalAccuracy(name = "acc")]
        super().compile(optimizer = opt, loss = self.loss_obj, metrics = mets, **kwargs)

    def train_step(self, data):
        graphs_batch, y_true = data
        y_true = tf.convert_to_tensor(y_true, dtype = tf.int32)
        with tf.GradientTape() as tape:
            if self.cfg.diffpool_mode == "batched":
                logits, lp, ent = self._forward_batched(graphs_batch, training = True)
            else:
                logits, lp, ent = self._forward_unbatched(graphs_batch, training = True)
            base_loss = self.loss_obj(y_true, logits)
            total_loss = base_loss + self.cfg.alpha_lp * lp + self.cfg.beta_ent * ent
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # update metrics
        for m in self.metrics:
            if not isinstance(m, keras.metrics.Mean):
                m.update_state(y_true, logits)
        
        self.loss_tracker.update_state(total_loss)
        self.base_loss_tracker.update_state(base_loss)
        self.lp_metric.update_state(lp)
        self.ent_metric.update_state(ent)
        # return logs
        logs = {
                    "loss": self.loss_tracker.result(), 
                    "base_loss": self.base_loss_tracker.result(), 
                    "lp": self.lp_metric.result(), 
                    "ent": self.ent_metric.result()
                }
        
        for m in self.metrics:
            if m.name not in logs:
                if not isinstance(m.result(), dict):
                    logs[m.name] = m.result()
                else:
                    actual_metrics = m.result()
                    for (k, v) in actual_metrics.items():
                       if k not in logs:
                           logs[k] = v
        return logs

    def test_step(self, data):
        graphs_batch, y_true = data
        y_true = tf.convert_to_tensor(y_true, dtype = tf.int32)
        if self.cfg.diffpool_mode == "batched":
            logits, lp, ent = self._forward_batched(graphs_batch, training = False)
        else:
            logits, lp, ent = self._forward_unbatched(graphs_batch, training = False)
        base_loss = self.loss_obj(y_true, logits)
        total_loss = base_loss + self.cfg.alpha_lp * lp + self.cfg.beta_ent * ent
        
        for m in self.metrics:
            if not isinstance(m, keras.metrics.Mean):
                m.update_state(y_true, logits)
        
        self.loss_tracker.update_state(total_loss)
        self.base_loss_tracker.update_state(base_loss)
        self.lp_metric.update_state(lp)
        self.ent_metric.update_state(ent)
        
        logs = {
                    "loss": self.loss_tracker.result(), 
                    "base_loss": self.base_loss_tracker.result(), 
                    "lp": self.lp_metric.result(), 
                    "ent": self.ent_metric.result()
                }
        
        for m in self.metrics:
            if m.name not in logs:
                if not isinstance(m.result(), dict):
                    logs[m.name] = m.result()
                else:
                    actual_metrics = m.result()
                    for (k, v) in actual_metrics.items():
                       if k not in logs:
                           logs[k] = v
        return logs

    def fit(self, train_graphs, y_train, val_graphs = None, y_val = None, epochs = None, **kwargs):
        train_seq = GraphBatchSequence(train_graphs, y_train, self.cfg.batch_size)
        val_seq = GraphBatchSequence(val_graphs, y_val, self.cfg.batch_size) if val_graphs is not None else None
        
        if self.cfg.logfile is not None:
            logfile = self._create_logfile(self.cfg.logfile)
        else:
            logfile = None
        
        for ep in range(epochs):
            print(f"Epoch {ep + 1} / {epochs}")
            pbar = trange(len(train_seq), ascii = ".>=", 
                          bar_format = "{n_fmt}|{bar}|{total_fmt} [ETA: {remaining}] | {postfix}",
                          ncols = 130)
            
            for batch_graphs in train_seq:
                metrics = self.train_step(batch_graphs)
                postfix = ' - '.join([f'{k}: {v.numpy():.4f}' for (k, v) in metrics.items() if k in {'loss', 'acc'}])
                pbar.set_postfix_str(postfix)
                pbar.update(1)
                
            self.reset_metrics()
            # self.compiled_metrics.reset_state()
            
            # Write to log file after each epoch, thus writing only the epoch-averaged metrics
            if logfile is not None:
                log_txt = f"Epoch {ep + 1}/{epochs}\n" + ' | '.join([f'{k}: {v.numpy():.4f}' for (k, v) in metrics.items()]) + ' | '
                with logfile.open('a', encoding = "utf-8") as f:
                    f.write(log_txt)
        
            if val_seq is not None:
                curr_postfix = None
                
                for batch_graphs in val_seq:
                    val_metrics = self.test_step(batch_graphs)                    
                
                postfix = ' - '.join([f'val_{k}: {v:.4f}' for (k, v) in val_metrics.items() if k in {'loss', 'acc'}])
                curr_postfix = pbar.postfix if curr_postfix is None else curr_postfix
                pbar.set_postfix_str(f"{curr_postfix} - {postfix}")
                
                # Write val metrics as well.
                if logfile is not None:
                    log_txt = ' | '.join([f'val_{k}: {v.numpy():.4f}' for (k, v) in val_metrics.items()]) + '\n'
                    with logfile.open('a', encoding = "utf-8") as f:
                        f.write(log_txt)
                        
                self.reset_metrics()
                # self.compiled_metrics.reset_state()
            
            pbar.close()

def dummy_TUGraphs(num_graphs):
    '''
    Helpful for smoke-test(s)
    '''
    num_feats = 128
    num_nodes = 100
    num_edges = 15
    y = 1
    
    tu_graph = TUGraph(np.random.random((num_nodes, num_feats)),
                       np.random.randint(0, num_nodes, size = (2, num_edges)),
                       y)
    
    return [tu_graph for _ in range(num_graphs)]


# ------------------------
# Train w/o CV / Evaluate
# ------------------------
def run_train(cfg: Config, root: str, name: str):
    graphs_tu, num_classes = TUDataset(root, name)
    # Convert to Graph objects and numpy labels
    graphs = [TUGraph_to_Graph(g) for g in graphs_tu]
    labels = np.array([int(g.y.numpy()) for g in graphs], dtype = np.int32)

    # Simple stratified split (75/25). You can swap to 10‑fold if desired.
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 42)
    (train_idx, val_idx), = sss.split(np.zeros_like(labels), labels)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    # Build trainer
    backbone = GCNDiffPool(num_classes, cfg)
    trainer = Trainer(backbone, cfg)
    trainer.compile(optimizer = optimizers.Adam(cfg.lr),
                    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
                    metrics = [keras.metrics.SparseCategoricalAccuracy(name = "acc")])

    # Shuffle once per run (fit will handle batching)
    perm = np.random.permutation(len(train_graphs))
    train_graphs = [train_graphs[i] for i in perm]
    y_train = y_train[perm]
    history = trainer.fit(train_graphs, y_train,
                          val_graphs = val_graphs, y_val = y_val,
                          epochs = cfg.epochs,
                          verbose = 1)
    return history

# --------
#   Main
# --------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type = Path, default = str(Path(__file__).resolve().parents[1] / "example_data"))
    p.add_argument("--name", type = str, default = "PROTEINS_full")
    p.add_argument("--hidden", type = int, default = 64)
    p.add_argument("--clusters", type = int, default = 25)
    p.add_argument("--dropout", type = float, default = 0.2)
    p.add_argument("--lr", type = float, default = 1e-3)
    p.add_argument("--epochs", type = int, default = 50)
    p.add_argument("--batch_size", type = int, default = 16)
    p.add_argument("--alpha_lp", type = float, default = 1.0)
    p.add_argument("--beta_ent", type = float, default = 0.1)
    p.add_argument("--pool", type = str, default = "sum", choices = ["sum", "mean"])
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--diffpool_mode", type = str, default = "batched", choices = ["batched", "unbatched"])
    p.add_argument("--logfile", type = Path, required = False)
    args = p.parse_args()

    cfg = Config(hidden = args.hidden, clusters = args.clusters, dropout = args.dropout,
                 lr = args.lr, epochs = args.epochs, batch_size = args.batch_size,
                 pool = args.pool, diffpool_mode = args.diffpool_mode,
                 alpha_lp = args.alpha_lp, beta_ent = args.beta_ent, logfile = args.logfile)
    
    for dev in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(dev, True)
    
    tf.config.run_functions_eagerly(True)

    fix_seeds(args.seed)
    run_train(cfg, args.root, args.name)

if __name__ == "__main__":
    main()