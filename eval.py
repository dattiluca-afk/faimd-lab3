import torch
import torch.nn as nn

# Import local modules
from models.model import CustomNet
from dataset.dataset import get_dataloaders

def validate(model, val_loader, criterion, device):
    """
    Evaluation loop to calculate accuracy on the validation set.
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # No gradients needed for evaluation
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    
    print(f'Final Evaluation -> Loss: {avg_loss:.6f} | Accuracy: {acc:.2f}%')
    return acc

def main():
    # 1. Setup Device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Data (Only the validation loader is needed for eval)
    _, val_loader = get_dataloaders(batch_size=64)

    # 3. Initialize Model and Load Trained Weights
    model = CustomNet().to(device)
    
    # This loads the "best_model.pth" file created during training
    try:
        model.load_state_dict(torch.load('model_best.pth', map_location=device))
        print("Successfully loaded trained model weights.")
    except FileNotFoundError:
        print("Error: 'model_best.pth' not found. Run train.py first to generate it.")
        return

    # 4. Define Criterion (for loss calculation)
    criterion = nn.CrossEntropyLoss()

    # 5. Run Evaluation
    validate(model, val_loader, criterion, device)

if __name__ == "__main__":
    main()