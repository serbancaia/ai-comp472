# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:51:14 2024

@author: TristanM2, Meliimoon, serbancaia
"""

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, ConcatDataset
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


k_folds = 10
num_epochs = 20
num_classes = 4
learning_rate = 0.0005

results = []

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define directories
dataset_dir = './ProjectDatasets'

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=np.random.seed(0))


class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        # CNN architecture
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

        )

        self.fc_layer = nn.Sequential(

            nn.Linear(in_features=8 * 8 * 128, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 4),

        )

    def forward(self, x):
        # Feeding image through convolutional and pooling layers
        x = self.conv_layer(x)

        # print('x_shape:',x.shape)

        # Flatten
        x = x.view(-1, 8 * 8 * 128)  # Flatten the tensor to a 1-D vector

        # Fully connected layer
        x = self.fc_layer(x)

        return x


if __name__ == "__main__":
    for fold, (train_val_set, test_set) in enumerate(kfold.split(dataset)):
        print(train_val_set)
        train_set, val_set = train_val_set[:85, :], train_val_set[85:, :]

        # Create the DataLoaders
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
        validation_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

        model = ConvNeuralNet()  # Creating an instance of the CNN
        model.apply(reset_weights)

        criterion = nn.CrossEntropyLoss()  # Includes SoftMax, so we do not need a SoftMax activation function at the end of the last fc layer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        total_steps = len(train_loader)

        best_val_loss = float('inf')
        patience = 6  # Number of epochs to wait before early stopping
        trigger_times = 0

        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backprop and optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Train accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                if (i + 1) % 10 == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy for fold {}: {:.2f}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                              total_steps, loss.item(), fold,
                                                                                              (correct / total) * 100))
                    results[fold] = 100.0 * (correct / total)

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            average_val_loss = val_loss / len(validation_loader)
            accuracy = 100 * correct / total

            print(f'Validation Loss: {average_val_loss}, Accuracy: {accuracy}%')

            # Early stopping
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                trigger_times = 0

                # Saving best-performing model (based on validation set)
                path = './main_kfold_best_model.pth'
                if os.path.isfile(
                        path):  # File exists, will compare best model with current model and will save the better model
                    # Define validation evaluation for the saved model
                    def current_saved_model_eval(model, dataloader, criterion):
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for images, labels in validation_loader:
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()

                        average_val_loss = val_loss / len(validation_loader)
                        return average_val_loss


                    saved_model = ConvNeuralNet()  # Model creation as instance of ConvNeuralNet
                    saved_model.load_state_dict(torch.load(path))  # Load saved model
                    saved_model_loss = current_saved_model_eval(saved_model, validation_loader,
                                                                criterion)  # Evaluate saved model

                    if average_val_loss < saved_model_loss:  # Compare saved model with current model, save current model as new best model, do nothing otherwise
                        torch.save(model.state_dict(), 'main_kfold_best_model.pth')
                        print("New best model saved.")

                else:  # File does not exist, first ever model will be saved
                    torch.save(model.state_dict(), 'main_kfold_best_model.pth')
                    print("First best model saved.")

            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        # Print fold results

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')
