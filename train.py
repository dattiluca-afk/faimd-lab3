import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import local modules
from models.model import CustomNet
from dataset.dataset import prepare_data, get_dataloaders

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f'\n--- Epoch {epoch} ---')

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to GPU/CPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Optimization steps
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch: [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    print(f'Train Results: Loss: {avg_loss:.6f} | Acc: {acc:.2f}%')
    return avg_loss, acc

def validate(model, val_loader, criterion, device):   
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    print(f'Validation Results: Loss: {avg_loss:.6f} | Acc: {acc:.2f}%')
    return acc

def main():
    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Prepare Data (Download and Format)
    prepare_data()
    train_loader, val_loader = get_dataloaders(batch_size=64)

    # 3. Initialize Model, Loss, Optimizer
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 4. Scheduler (Based on your Colab logic)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        # Train
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_acc = validate(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step(val_acc)

        # Save the best model
        if val_acc > best_acc:
            print(f'New Best Accuracy! Saving model...')
            best_acc = val_acc
            torch.save(model.state_dict(), 'model_best.pth')

    print(f'\nTraining complete. Best Validation Accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    main()