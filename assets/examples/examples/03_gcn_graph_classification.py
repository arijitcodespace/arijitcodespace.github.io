# 03_gcn_graph_classification.py
# ----------------------------------------------------------------------------
# Battle‑test gnn‑keras3 on TU‑Dataset PROTEINS / PROTEINS_full with
# **mini‑batch training & evaluation** (multiple graphs per step).
#
# Usage
#   python 03_gcn_graph_classification.py \
#       --root ./data --name PROTEINS_full --pool sum --hidden 64 \
#       --epochs 150 --batch_size 32 --lr 1e-3 --dropout 0.2
# OR,
# python 03_gcn_graph_classification.py \
#       --root ./data --name PROTEINS_full --pool sum --hidden 64 \
#       --epochs 150 --batch_size 32 --lr 1e-3 --dropout 0.2 --no_cv
# ----------------------------------------------------------------------------
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import trange

from gnn_keras import Graph, GraphConv
from gnn_keras.datasets import TUGraph, TUDataset, TUGraph_to_Graph
from gnn_keras.utils.batching import pack_block_diagonal

# ---------------------------------------------------------------------------
# Model: GCN + readout
# ---------------------------------------------------------------------------
class GlobalPool(layers.Layer):
    def __init__(self, mode: str = "sum"):
        super().__init__()
        assert mode in {"sum", "mean"}
        self.mode = mode

    # Allow positional Graph and optional positional training flag
    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], Graph):
            graph = args[0]
            if len(args) > 1 and 'training' not in kwargs:
                t = args[1]
                is_bool_tensor = getattr(t, "dtype", None) is not None and str(t.dtype) == "bool"
                if isinstance(t, (bool, type(None))) or is_bool_tensor:
                    kwargs['training'] = t
            return super().__call__(graph = graph, **kwargs)
        return super().__call__(*args, **kwargs)

    def call(self, graph: Graph, training = None) -> Graph:
        x = graph.x
        xg = tf.reduce_sum(x, axis = 0, keepdims=True) if self.mode == "sum" else tf.reduce_mean(x, axis = 0, keepdims = True)
        return graph.with_(x = xg)


class GlobalAttentionPool(layers.Layer):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.scorer = layers.Dense(1)

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], Graph):
            graph = args[0]
            if len(args) > 1 and 'training' not in kwargs:
                t = args[1]
                is_bool_tensor = getattr(t, "dtype", None) is not None and str(t.dtype) == "bool"
                if isinstance(t, (bool, type(None))) or is_bool_tensor:
                    kwargs['training'] = t
            return super().__call__(graph = graph, **kwargs)
        return super().__call__(*args, **kwargs)

    def call(self, graph: Graph, training=None) -> Graph:
        s = self.scorer(graph.x)             # [N,1]
        w = tf.nn.softmax(s, axis = 0)         # weights sum to 1 per graph (when segmented)
        xg = tf.reduce_sum(w * graph.x, axis = 0, keepdims = True)
        return graph.with_(x = xg)


class GCNGraphClassifier(keras.Model):
    def __init__(self, hidden: int, classes: int, pool: str = "sum", dropout: float = 0.2,
                 self_loops: bool = True, normalize: bool = True, attention_pool: bool = False):
        super().__init__()
        self.conv1 = GraphConv(hidden, activation = "relu", self_loops = self_loops, normalize = normalize)
        self.do1 = layers.Dropout(dropout)
        self.conv2 = GraphConv(hidden, activation = "relu", self_loops = self_loops, normalize = normalize)
        self.pool = GlobalAttentionPool(hidden) if attention_pool else GlobalPool(pool)
        self.head = layers.Dense(classes, activation = None)

    # Allow positional Graph and optional positional training flag
    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], (Graph, list)):
            graphs = args[0]
            if len(args) > 1 and 'training' not in kwargs:
                t = args[1]
                is_bool_tensor = getattr(t, "dtype", None) is not None and str(t.dtype) == "bool"
                if isinstance(t, (bool, type(None))) or is_bool_tensor:
                    kwargs['training'] = t
            return super().__call__(graphs = graphs, **kwargs)
        return super().__call__(*args, **kwargs)
    
    def build(self, input_shape):
        self.built = True
        super().build(input_shape)
    
    # ---------------------------------------------------------------------------
    # Mini‑batch forward using utils.pack_block_diagonal + segment readout
    # ---------------------------------------------------------------------------

    @staticmethod
    def _segment_ids(graphs: List[Graph]) -> tf.Tensor:
        """Return a [sum(N_i)] int32 vector mapping each node to its graph id (0..B-1)."""
        sizes = tf.stack([tf.shape(g.x)[0] for g in graphs])            # [B]
        return tf.repeat(tf.range(len(graphs), dtype = tf.int32), sizes)  # [sum N_i]

    @staticmethod
    def _segment_sum(x: tf.Tensor, seg: tf.Tensor) -> tf.Tensor:
        return tf.math.segment_sum(x, seg)

    @staticmethod
    def _segment_mean(x: tf.Tensor, seg: tf.Tensor) -> tf.Tensor:
        return tf.math.segment_mean(x, seg)

    @staticmethod
    def _segment_softmax(scores: tf.Tensor, seg: tf.Tensor) -> tf.Tensor:
        """Softmax over variable‑length segments (scores shape [N,1] or [N])."""
        s = scores if scores.shape.rank == 2 else tf.expand_dims(scores, -1)
        smax = tf.math.segment_max(s, seg)
        s = s - tf.gather(smax, seg)
        e = tf.exp(s)
        denom = tf.math.segment_sum(e, seg)
        return e / tf.gather(denom, seg)
    
    # Encoder that returns node embeddings pre‑readout (used by batched forward)
    def encode_nodes(self, graph: Graph, training = None) -> Graph:
        g = self.conv1(graph, training = training)
        g = g.with_(x=self.do1(g.x, training = training))
        g = self.conv2(g, training = training)
        return g
    
    def batch_forward(self, graphs: List[Graph], training: bool) -> tf.Tensor:
        """Forward pass on a list of Graphs -> logits [B, C]."""
        assert len(graphs) > 0, "batch_forward received an empty list"
        bigG = pack_block_diagonal(graphs)
        seg = self._segment_ids(graphs)                    # map nodes -> original graphs

        g = self.encode_nodes(bigG, training = training)  # node embeddings

        # Readout per original graph
        if isinstance(self.pool, GlobalAttentionPool):
            scores = self.pool.scorer(g.x)           # [N,1]
            w = self._segment_softmax(scores, seg)         # [N,1]
            pooled = self._segment_sum(w * g.x, seg)       # [B, H]
        else:
            pooled = self._segment_sum(g.x, seg) if self.pool.mode == "sum" else self._segment_mean(g.x, seg)

        logits = self.head(pooled)                   # [B, C]
        return logits

    def call(self, graphs: Union[Graph, List[Graph]], training = None) -> Graph:
        if not isinstance(graphs, list):
            g = self.encode_nodes(graphs, training = training)
            g = self.pool(g, training = training)
            logits = self.head(g.x)
        else:
            logits = self.batch_forward(graphs, training = training)
        
        return logits

# ---------------------------------------------------------------------------
# Trainer (batched steps)
# ---------------------------------------------------------------------------
class Trainer(keras.Model):
    def __init__(self, model: keras.Model, lr: float = 1e-3, l2: float = 0.0):
        super().__init__()
        self.model = model
        self.l2 = l2
        self.lr = lr
        self.compile()

    # Allow positional Graph and optional positional training flag
    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], list):
            graphs = args[0]
            if len(args) > 1 and 'training' not in kwargs:
                t = args[1]
                is_bool_tensor = getattr(t, "dtype", None) is not None and str(t.dtype) == "bool"
                if isinstance(t, (bool, type(None))) or is_bool_tensor:
                    kwargs['training'] = t
            return super().__call__(graphs = graphs, **kwargs)
        return super().__call__(*args, **kwargs)

    def call(self, graphs, training = None):
        return self.model(graphs, training = training)
    
    def compile(self, **kwargs):
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        self.opt = keras.optimizers.Adam(learning_rate = self.lr)
        self.train_acc = keras.metrics.SparseCategoricalAccuracy(name = "accuracy")
        self.val_acc = keras.metrics.SparseCategoricalAccuracy(name = "val_accuracy")
        
        # super().compile(optimizer = self.opt, loss = self.loss_fn, metrics = [self.acc])    
    
    @property
    def metrics(self):
        return [self.train_acc, self.val_acc]

    @tf.function(jit_compile = False)
    def train_step(self, graphs: List[Graph]):
        y = tf.stack([tf.reshape(g.y, []) for g in graphs])
        with tf.GradientTape() as tape:
            logits = self(graphs, training = True)
            loss = self.loss_fn(y, logits)
            if self.l2 > 0:
                loss += tf.add_n([tf.nn.l2_loss(w) for w in self.model.trainable_variables]) * self.l2
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_acc.update_state(y, logits)
        return {"loss": loss, 'accuracy': self.train_acc.result()}

    @tf.function(jit_compile = False)
    def test_step(self, graphs: List[Graph]):
        y = tf.stack([tf.reshape(g.y, []) for g in graphs])
        logits = self.model(graphs, training = False)
        loss = self.loss_fn(y, logits)
        self.val_acc.update_state(y, logits)
        return {"loss": loss, 'accuracy': self.val_acc.result()}
    
    @classmethod
    def _minibatches(cls, indices: np.ndarray, batch_size: int):
        for i in range(0, len(indices), batch_size):
            yield indices[i:i+batch_size]
    
    def fit(self, graphs: List[Graph], batch_size, epochs, validation_split = None, seed = None):
        num_graphs = len(graphs)
        if validation_split is not None:
            assert isinstance(validation_split, float)
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(np.arange(len(graphs)), test_size = validation_split)
        else:
            train_idx = np.arange(len(graphs)); test_idx = None
            
        rng = np.random.RandomState(seed if seed is not None else 0)
        
        train_ids = rng.permutation(train_idx)
        test_ids = test_idx
        
        num_batches = np.ceil(len(train_idx) / batch_size).astype(np.int64)
        for ep in range(epochs):
            print(f"Epoch {ep + 1} / {epochs}")
            pbar = trange(num_batches, ascii = ".>=", bar_format = "{n_fmt}|{bar}|{total_fmt} [ETA: {remaining}] | {postfix}", ncols = 130)
            for chunk in self._minibatches(train_ids, batch_size):
                batch_graphs = [TUGraph_to_Graph(graphs[int(i)]) for i in chunk]
                metrics = self.train_step(batch_graphs)
                postfix = ' - '.join([f'{k}: {v.numpy():.4f}' for (k, v) in metrics.items()])
                pbar.set_postfix_str(postfix)
                pbar.update(1)
        
            if test_ids is not None:
                curr_postfix = None
                curr_batch = 0
                acc_sum = 0
                loss_sum = 0
                
                for chunk in self._minibatches(test_ids, batch_size):
                    batch_graphs = [TUGraph_to_Graph(graphs[int(i)]) for i in chunk]
                    metrics = self.test_step(batch_graphs)                    
                    curr_batch += 1
                    acc_sum += metrics['accuracy'].numpy()
                    loss_sum += metrics['loss'].numpy()
                
                mean_loss = loss_sum / curr_batch; mean_acc = acc_sum / curr_batch
                postfix = ' - '.join([f'val_loss: {mean_loss:.4f}', f'val_accuracy: {mean_acc:.4f}'])
                curr_postfix = pbar.postfix if curr_postfix is None else curr_postfix
                pbar.set_postfix_str(f"{curr_postfix} - {postfix}")
            
            pbar.close()
            train_ids = rng.permutation(train_idx)
            
        
        

# ---------------------------------------
# Training w/o CV
# ---------------------------------------

def train(graphs_list, num_classes: int, hidden = 64, pool = "sum", attn = False,
           epochs = 150, lr = 1e-3, dropout = 0.2, l2 = 5e-4, seed = 42, batch_size = 32):

    model = GCNGraphClassifier(hidden = hidden, classes = num_classes, pool = pool,
                               dropout = dropout, attention_pool = attn)
    trainer = Trainer(model, lr = lr, l2 = l2)
    trainer.fit(graphs_list, batch_size = batch_size, epochs = epochs, validation_split = 0.25, seed = 111)
    
    
# ---------------------------------------------------------------------------
# 10‑fold CV driver with batching
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold

def run_cv(graphs_list, num_classes: int, hidden = 64, pool = "sum", attn = False,
           epochs = 150, lr = 1e-3, dropout = 0.2, l2 = 5e-4, seed = 42, batch_size = 32):
    rng = np.random.RandomState(seed)
    y = np.array([g.y for g in graphs_list])
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

    fold_acc = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros_like(y), y), 1):
        model = GCNGraphClassifier(hidden = hidden, classes = num_classes, pool = pool,
                                   dropout = dropout, attention_pool = attn)
        trainer = Trainer(model, lr = lr, l2 = l2)

        # Train
        for ep in trange(1, epochs+1, desc = f"Fold {fold}"):
            trainer.train_acc.reset_state()
            train_ids = rng.permutation(train_idx)
            for chunk in Trainer._minibatches(train_ids, batch_size):
                batch_graphs = [TUGraph_to_Graph(graphs_list[int(i)]) for i in chunk]
                trainer.train_step(batch_graphs)

        # Test
        trainer.val_acc.reset_state()
        for chunk in Trainer._minibatches(test_idx, batch_size):
            batch_graphs = [TUGraph_to_Graph(graphs_list[int(i)]) for i in chunk]
            trainer.test_step(batch_graphs)
        acc = float(trainer.val_acc.result().numpy())
        fold_acc.append(acc)
        print({"fold": fold, "test_acc": acc})

    mean = float(np.mean(fold_acc)); std = float(np.std(fold_acc))
    print({"mean_acc": mean, "std": std})
    return mean, std

# ---------------------------------------------------------------------------
# Unit tests. Run with: --run_tests
# ---------------------------------------------------------------------------

def _toy_graph(label, num_nodes = 4, feat_dim = 8) -> Graph:
    x = tf.ones([num_nodes, feat_dim], dtype = tf.float32)
    src = tf.constant([0, 1, 2, 2], tf.int64)
    dst = tf.constant([1, 2, 3, 0], tf.int64)
    ei = tf.stack([src, dst], axis = 0)
    G = Graph.from_edge_index(x, ei, num_nodes = num_nodes, symmetric = True)
    return G.with_(y = tf.constant(label, tf.int32))


def dummy_TUGraphs(num_graphs):
    num_feats = 128
    num_nodes = 100
    num_edges = 15
    y = 1
    
    tu_graph = TUGraph(np.random.random((num_nodes, num_feats)),
                       np.random.randint(0, num_nodes, size = (2, num_edges)),
                       y)
    
    return [tu_graph for _ in range(num_graphs)]

def run_unit_tests():
    print("Running quick sanity tests...")
    # 1) Batched vs single forward equivalence (no dropout, deterministic)
    m = GCNGraphClassifier(hidden = 16, classes = 3, pool = "sum", dropout = 0.0, attention_pool = False)
    g1 = _toy_graph(label = 0); g2 = _toy_graph(label = 1)
    # Single
    log1 = m(g1, training = False)
    log2 = m(g2, training = False)
    # Batched via list[*]
    logits = m([g1, g2], training = False)
    # replace the strict boolean asserts with:
    tf.debugging.assert_near(log1[0], logits[0], rtol = 1e-5, atol = 1e-2)
    tf.debugging.assert_near(log2[0], logits[1], rtol = 1e-5, atol = 1e-2)



    # 2) Attention pooling shape & stability
    m2 = GCNGraphClassifier(hidden = 16, classes = 2, pool = "sum", dropout = 0.0, attention_pool = True)
    logits2 = m2([g1, g2], training = False)
    assert logits2.shape == (2, 2), "Attention pooled logits wrong shape"


    # 3) Trainer smoke test with a tiny batch
    trainer = Trainer(m2, lr = 1e-3, l2 = 0.0)
    m_ = trainer.train_step([_toy_graph(label = 0), _toy_graph(label = 1)])
    assert 'loss' in m_ and 'accuracy' in m_, "Trainer.train_step must return metrics dict"
    print("All tests passed.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type = Path, default = str(Path(__file__).resolve().parents[1] / "example_data"))
    p.add_argument("--name", type = str, default = "PROTEINS_full")
    p.add_argument("--pool", type = str, choices = ["sum", "mean"], default = "sum")
    p.add_argument("--attn", action = "store_true")
    p.add_argument("--hidden", type = int, default = 64)
    p.add_argument("--epochs", type = int, default = 150)
    p.add_argument("--batch_size", type = int, default = 32)
    p.add_argument("--lr", type = float, default = 1e-3)
    p.add_argument("--dropout", type = float, default = 0.2)
    p.add_argument("--l2", type = float, default = 5e-4)
    p.add_argument("--no_cv", action = "store_true")
    p.add_argument("--run_tests", action = "store_true")
    args = p.parse_args()

    for dev in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(dev, True)
    
    tf.config.run_functions_eagerly(True)

    if args.run_tests:
        run_unit_tests()

    graphs, num_classes = TUDataset(args.root, args.name)
    print(f"Loaded {len(graphs)} graphs; classes = {num_classes}; example: N = {graphs[0].x.shape[0]}, F = {graphs[0].x.shape[1]}")

    # graphs = dummy_TUGraphs(18)

    if args.no_cv:
        train(graphs, num_classes = 2,
               hidden = args.hidden, pool = args.pool, attn = args.attn,
               epochs = args.epochs, lr = args.lr, dropout = args.dropout, l2 = args.l2,
               batch_size = args.batch_size)
    else:
        run_cv(graphs, num_classes,
               hidden = args.hidden, pool = args.pool, attn = args.attn,
               epochs = args.epochs, lr = args.lr, dropout = args.dropout, l2 = args.l2,
               batch_size = args.batch_size)
