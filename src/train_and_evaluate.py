import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Net
from data_loader import load_data_cifar10

def train_and_evaluate_model(model, train_iter, test_iter, criterion, optimizer, num_epochs=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_losses = []
    epoch_train_acc = []
    epoch_test_acc = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            batch_losses.append(loss.item())

        train_accuracy = 100 * correct_train / total_train
        epoch_train_acc.append(train_accuracy)

        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_iter:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        epoch_test_acc.append(test_accuracy)

        print(f'Epoch {epoch + 1}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%')

    print('Finished Training and Evaluation')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, label='Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Batch Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_train_acc, label='Training Accuracy', color='blue')
    plt.plot(epoch_test_acc, label='Testing Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()

    plt.show()


model = Net(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=10e-6)


train_and_evaluate_model(model, train_iter, test_iter, criterion, optimizer)


if __name__ == '__main__':
    train_and_evaluate_model()