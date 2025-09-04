import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from dataset import get_dataloaders
from model import get_resnet_model
from utils import save_model
from config import device, epochs, lr, model_save_path
from evaluate import evaluate_model


def main():
    # Load data
    train_loader, test_loader, label2idx = get_dataloaders()
    class_names = list(label2idx.keys())
    num_classes = len(class_names)

    # Initialize model, loss, optimizer
    model = get_resnet_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"→ Training on {len(train_loader.dataset)} samples for {epochs} epochs")
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the model
    print("\n→ Evaluating on test set")
    acc, report = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(report)

if __name__ == '__main__':
    main()
