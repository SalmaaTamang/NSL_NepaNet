#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set CUDNN benchmark for optimization
torch.backends.cudnn.benchmark = True

# Image transformations for train, validation, and test sets
image_transform = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
}

# Dataset directories
dataset = 'Data'
train_dir = os.path.join(dataset, 'train')
valid_dir = os.path.join(dataset, 'valid')
test_dir = os.path.join(dataset, 'test')

# Load datasets
data = {
    'train': datasets.ImageFolder(root=train_dir, transform=image_transform['train']),
    'valid': datasets.ImageFolder(root=valid_dir, transform=image_transform['valid']),
    'test': datasets.ImageFolder(root=test_dir, transform=image_transform['test'])
}

# Loaders
train_data_loader = DataLoader(data['train'], batch_size=16, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=16)
test_data_loader = DataLoader(data['test'], batch_size=16)

# Dataset sizes
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# Custom NSLNet model
class NslNet(nn.Module):
    def __init__(self, num_classes):
        super(NslNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),  # Grayscale input
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

# Initialize models
def initialize_model(model_name, num_classes):
    if model_name == 'alexnet':
        model = models.alexnet(weights='DEFAULT')
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'nepanet':
        model = NslNet(num_classes)
    else:
        raise ValueError("Model not supported.")
    
    return model

# Define loss function and optimizer
loss_criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train_valid(model, loss_criterion, optimizer, device, epochs=30):
    start_time = time.time()
    history = []

    model.to(device)

    for epoch in range(epochs):
        print('Epoch: {}/{}'.format(epoch + 1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        # Training phase
        for inputs, labels in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
            epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))

    print("Training complete. Total time: {:.4f}s".format(time.time() - start_time))
    
    return model, history

# Function to evaluate test accuracy and generate confusion matrix
def evaluate_test(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_test_loss = test_loss / test_data_size
    test_acc = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return avg_test_loss, test_acc, cm

# Models to compare
models_to_compare = ['alexnet', 'resnet50', 'nepanet']
num_classes = len(os.listdir(train_dir))
history_dict = {model_name: [] for model_name in models_to_compare}

# Train the models
num_epochs = 25
for model_name in models_to_compare:
    print(f"Training {model_name}...")
    model = initialize_model(model_name, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trained_model, history = train_valid(model, loss_criterion, optimizer, device, num_epochs)
    history_dict[model_name] = history

    # Evaluate test set
    avg_test_loss, test_accuracy, confusion_mat = evaluate_test(trained_model, test_data_loader, device)
    
    # Print test results
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_mat, display_labels=data['train'].classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Convert history dictionary to numpy arrays for easier plotting
train_losses = np.array([np.array(history)[:, 0] for history in history_dict.values()])
valid_losses = np.array([np.array(history)[:, 1] for history in history_dict.values()])
train_accuracies = np.array([np.array(history)[:, 2] for history in history_dict.values()])
valid_accuracies = np.array([np.array(history)[:, 3] for history in history_dict.values()])

# Plotting Train Loss Comparison
plt.figure(figsize=(12, 6))
for i, model_name in enumerate(models_to_compare):
    plt.plot(train_losses[i], label=f'{model_name} Training Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Plotting Validation Loss Comparison
plt.figure(figsize=(12, 6))
for i, model_name in enumerate(models_to_compare):
    plt.plot(valid_losses[i], label=f'{model_name} Validation Loss')
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Plotting Train Accuracy Comparison
plt.figure(figsize=(12, 6))
for i, model_name in enumerate(models_to_compare):
    plt.plot(train_accuracies[i], label=f'{model_name} Training Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.show()

# Plotting Validation Accuracy Comparison
plt.figure(figsize=(12, 6))
for i, model_name in enumerate(models_to_compare):
    plt.plot(valid_accuracies[i], label=f'{model_name} Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.show()

