# ============================================================
# Differential Privacy vs Non-DP SGD Comparison on MNIST
# ============================================================

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from opacus import PrivacyEngine
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
EPOCHS = 5
BATCH_SIZE = 128
LR = 0.1
MAX_GRAD_NORM = 1.0
TARGET_DELTA = 1e-5

# Dynamically generated epsilon values (modify as needed)
EPSILONS = list(range(1, 26, 4))   # Example: 1,5,9,13,17,21,25

# Storage
results = {
    "non_dp": {"acc": [], "loss": []},
    "dp": {}
}

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=False, download=True, transform=transform),
    batch_size=1024,
    shuffle=False
)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 28 * 28)
            preds = model(x)
            test_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total, test_loss / len(test_loader)


# -----------------------------
# MODEL FACTORY
# -----------------------------
def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# -----------------------------
# 1. TRAIN NON-DP BASELINE
# -----------------------------
model = create_model()
optimizer = optim.SGD(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

print("\nTraining Non-DP SGD")

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

    acc, test_loss = evaluate(model)
    results["non_dp"]["acc"].append(acc)
    results["non_dp"]["loss"].append(test_loss)

    print(f"Epoch {epoch+1}: acc={acc:.4f} loss={test_loss:.4f}")

# -----------------------------
# 2. TRAIN DP-SGD MODELS
# -----------------------------
for epsilon in EPSILONS:

    print(f"\nTraining DP-SGD with target epsilon={epsilon}")

    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    privacy_engine = PrivacyEngine()

    model, optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=TARGET_DELTA,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM
    )

    loss_fn = nn.CrossEntropyLoss()

    acc_list = []
    loss_list = []

    for epoch in range(EPOCHS):
        model.train()
        for x, y in dp_loader:
            x = x.view(-1, 28 * 28)
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        acc, test_loss = evaluate(model)
        acc_list.append(acc)
        loss_list.append(test_loss)

        print(f"Epoch {epoch+1}: acc={acc:.4f} loss={test_loss:.4f}")

    results["dp"][epsilon] = {
        "acc": acc_list,
        "loss": loss_list
    }

# ============================================================
# MODEL COMPARISON ENGINE
# ============================================================

def compare_models(results_dict):

    comparison = []

    # Non-DP baseline
    comparison.append({
        "model_type": "Non-DP",
        "epsilon": None,
        "final_acc": results_dict["non_dp"]["acc"][-1],
        "final_loss": results_dict["non_dp"]["loss"][-1]
    })

    # DP models (automatic)
    for eps, metrics in results_dict["dp"].items():
        comparison.append({
            "model_type": "DP",
            "epsilon": eps,
            "final_acc": metrics["acc"][-1],
            "final_loss": metrics["loss"][-1]
        })

    return comparison


def analyze_models(comparison, loss_threshold=None):

    print("\n==============================")
    print("FINAL MODEL COMPARISON")
    print("==============================")

    for m in comparison:
        print(m)

    # Maximum Accuracy
    best_acc = max(comparison, key=lambda x: x["final_acc"])
    print("\nModel with Maximum Accuracy:")
    print(best_acc)

    # Minimum Loss
    best_loss = min(comparison, key=lambda x: x["final_loss"])
    print("\nModel with Minimum Loss:")
    print(best_loss)

    # Optional constraint
    if loss_threshold is not None:
        filtered = [m for m in comparison if m["final_loss"] <= loss_threshold]

        if filtered:
            best_under_constraint = max(filtered, key=lambda x: x["final_acc"])
            print(f"\nBest Accuracy with Loss <= {loss_threshold}:")
            print(best_under_constraint)
        else:
            print(f"\nNo model satisfies loss <= {loss_threshold}")

    # Ranking
    ranked = sorted(comparison, key=lambda x: x["final_acc"], reverse=True)

    print("\nModels Ranked by Accuracy:")
    for r in ranked:
        print(r)

    return {
        "best_accuracy": best_acc,
        "best_loss": best_loss,
        "ranked": ranked
    }


# -----------------------------
# RUN COMPARISON
# -----------------------------
comparison_results = compare_models(results)

analysis = analyze_models(
    comparison_results,
    loss_threshold=0.30  # Set to None if no constraint needed
)