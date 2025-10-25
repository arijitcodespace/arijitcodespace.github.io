import tensorflow as tf
from gnn_keras.datasets import ToyCitation
from gnn_keras.models import GCNClassifier
from gnn_keras.trainers import NodeClassificationTrainer

G = ToyCitation(n_nodes=1000, n_feats=64, n_edges=4000, n_classes=5).graph()
model = GCNClassifier(hidden=64, classes=5)
trainer = NodeClassificationTrainer(model)

for epoch in range(3):
    metrics = trainer.train_step(G)
    print({"epoch": epoch, **{k: float(v) for k, v in metrics.items()}})
