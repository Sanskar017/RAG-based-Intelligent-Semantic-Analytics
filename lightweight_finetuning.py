import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# EPOCHS
# -----------------------------------

epochs = [1, 2, 3, 4, 5]

# -----------------------------------
# SIMULATED LOSSES
# -----------------------------------

train_loss = [
    1.25,
    0.92,
    0.71,
    0.54,
    0.41
]

val_loss = [
    1.31,
    1.02,
    0.83,
    0.73,
    0.68
]

# -----------------------------------
# ACCURACY
# -----------------------------------

train_acc = [
    0.58,
    0.69,
    0.77,
    0.84,
    0.90
]

val_acc = [
    0.54,
    0.65,
    0.72,
    0.78,
    0.81
]

# -----------------------------------
# TRAIN / VAL LOSS PLOT
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    marker='o',
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker='o',
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/training_validation_loss.png"
)

plt.close()

# -----------------------------------
# ACCURACY PLOT
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_acc,
    marker='o',
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_acc,
    marker='o',
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/training_validation_accuracy.png"
)

plt.close()

print(
    "Saved training_validation_loss.png"
)

print(
    "Saved training_validation_accuracy.png"
)