# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:51:14 2024

@author: TristanM2, Meliimoon, serbancaia,
"""

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import confusion_matrix_analysis_and_metrics_KFold_Model as confusion_matrix_calc


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


k_folds = 10
num_epochs = 20
num_classes = 4
learning_rate = 0.0005

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

random_state_instance = int(np.random.rand(1)[0]*100)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state_instance)


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
        # Flatten
        x = x.view(-1, 8 * 8 * 128)  # Flatten the tensor to a 1-D vector
        # Fully connected layer
        x = self.fc_layer(x)
        return x


if __name__ == "__main__":
    for z in range(0, 5):
        performance_metrics_tabular = [
            ["Fold", "Macro-Precision", "Macro-Recall", "Macro-F1", "Micro-Precision", "Micro-Recall", "Micro-F1",
             "Accuracy"], [], [], [], [], [], [], [], [], [], [], []]

        for fold, (train_val_indices, test_indices) in enumerate(kfold.split(dataset)):
            try:
                os.remove(f'./main_kfold_best_model_fold{fold+1}.pth')
            except OSError:
                pass

            train_val_subset = Subset(dataset, train_val_indices)
            test_subset = Subset(dataset, test_indices)

            # Further split train_val_subset into train and val
            train_size = int(0.85 * len(train_val_subset))
            val_size = len(train_val_subset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(train_val_subset, [train_size, val_size])

            # Create the DataLoaders
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
            validation_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

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
                        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}, Accuracy for fold {fold + 1}: {(correct / total) * 100:.2f}%')

                # Validation loop
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in validation_loader:
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
                    path = f'./main_kfold_best_model_fold{fold+1}.pth'
                    if os.path.isfile(path):  # File exists, will compare best model with current model and will save the better model
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
                        saved_model_loss = current_saved_model_eval(saved_model, validation_loader, criterion)  # Evaluate saved model

                        if average_val_loss < saved_model_loss:  # Compare saved model with current model, save current model as new best model, do nothing otherwise
                            torch.save(model.state_dict(), f'main_kfold_best_model_fold{fold+1}.pth')
                            print("New best model saved.")
                    else:  # File does not exist, first ever model will be saved
                        torch.save(model.state_dict(), f'main_kfold_best_model_fold{fold+1}.pth')
                        print("First best model saved.")
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        break
            performance_metrics_tabular[fold + 1] = confusion_matrix_calc.main(fold+1, test_loader)

        with open("./K-Fold Cross-Validation Performance Metrics old dataset.txt", 'a') as outputFile:
            print("K-Fold Cross-Validation Performance Metrics:")
            outputFile.write("K-Fold Cross-Validation Performance Metrics:\n")

            performance_metrics_tabular[len(performance_metrics_tabular) - 1].append("Average")

            for column in range(1, len(performance_metrics_tabular[0])):
                metric = 0.0
                for row in range(1, len(performance_metrics_tabular) - 1):
                    metric += (float(performance_metrics_tabular[row][column][:-1])/100)
                performance_metrics_tabular[len(performance_metrics_tabular) - 1].append(f"{((metric/(len(performance_metrics_tabular) - 2))*100):.4f}%")

            for row in performance_metrics_tabular:
                print(row)
                outputFile.write(f"{str(row)}\n")
            outputFile.write(f"\n")
