import tensorflow as tf
from gnn_keras.datasets import ToyCitation
from gnn_keras.models import GATClassifier
from gnn_keras.trainers import NodeClassificationTrainer

G = ToyCitation(n_nodes=800, n_feats=32, n_edges=3000, n_classes=3).graph()
model = GATClassifier(hidden=16, classes=3, heads=4)
trainer = NodeClassificationTrainer(model, lr=5e-3)

for epoch in range(3):
    metrics = trainer.train_step(G)
    print({"epoch": epoch, **{k: float(v) for k, v in metrics.items()}})
