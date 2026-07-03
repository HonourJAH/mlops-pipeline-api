"""Quick isolated test of train_model() before wiring into FastAPI.
Run this directly: python3 scripts/test_training.py
"""

from app.services.training import train_model

print("Starting training run...")

result = train_model(
    C=10.0,
    max_features=50000,
    ngram_range=(1, 1),
    max_iter=1000,
)

print("\n── Results ──────────────────────────────")
print(f"Run ID:            {result['run_id']}")
print(f"Training accuracy: {result['training_accuracy']}")
print(f"Test accuracy:     {result['test_accuracy']}")
print(f"Params:            {result['params']}")
print("\nCheck http://localhost:5000 to see this run in the MLflow UI")
