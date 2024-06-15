# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:08:46 2024

@author: SerbanCaia, Meliimoon, TristanM2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torchvision import datasets
from torch.utils.data import DataLoader


class ConvNeuralNet(nn.Module):
    def __init__(self, conv_dropout, fc_dropout, dropout_iterate, conv_layer_output, conv_layer_count,
                 fc_layer_count, max_pool_iterate,
                 kernel_size, stride_length, conv_output_size_iterate):
        super(ConvNeuralNet, self).__init__()
        layers = []
        image_size = 48
        power2_track = 0
        for j in range(0, conv_layer_count):
            if j == 0:
                if int((image_size - kernel_size) / stride_length + 1) < 4:
                    break
                layers.append(nn.Conv2d(1, conv_layer_output * (2 ** power2_track), kernel_size, stride_length))
                image_size = int((image_size - kernel_size) / stride_length + 1)
                layers.append(nn.BatchNorm2d(conv_layer_output * (2 ** power2_track)))
                layers.append(nn.LeakyReLU(True))
            else:
                if int((image_size - kernel_size) / stride_length + 1) < 4:
                    break
                layers.append(nn.Conv2d(conv_layer_output * (2 ** power2_track),
                                        conv_layer_output * (
                                                2 ** (power2_track + int(j % conv_output_size_iterate == 0))),
                                        kernel_size, stride_length))
                power2_track += int(j % conv_output_size_iterate == 0)
                image_size = int((image_size - kernel_size) / stride_length + 1)
                layers.append(nn.BatchNorm2d(conv_layer_output * (2 ** power2_track)))
                layers.append(nn.LeakyReLU(True))
            if (j + 1) % max_pool_iterate == 0:
                if int(image_size / 2) < 4:
                    break
                layers.append(nn.MaxPool2d(2, 2))
                image_size = int(image_size / 2)
                layers.append(nn.Dropout(conv_dropout))
            elif j == conv_layer_count - 1:
                if int(image_size / 2) < 4:
                    break
                layers.append(nn.MaxPool2d(2, 2))
                image_size = int(image_size / 2)
                layers.append(nn.Dropout(conv_dropout))

        self.conv_layer = nn.Sequential(*layers)

        fc_input1 = image_size * image_size * conv_layer_output * (2 ** power2_track)
        self.fc_input = fc_input1
        k = 0
        fc_output1 = 1
        while True:
            if fc_output1 * (2 ** k) >= fc_input1:
                break
            else:
                fc_output1 *= (2 ** k)
                k += 1

        fc_layers = []
        for j in range(0, fc_layer_count):
            if j == 0 and j == fc_layer_count - 1:
                fc_layers.append(nn.Linear(fc_input1, 4))
            elif j == 0:
                fc_layers.append(nn.Linear(fc_input1, fc_output1))
                fc_layers.append(nn.ReLU(True))
                fc_layers.append(nn.Dropout(fc_dropout))
            elif j == fc_layer_count - 1:
                fc_layers.append(nn.Linear(fc_output1, 4))
            else:
                fc_layers.append(nn.Linear(fc_output1, int(fc_output1 / 2)))
                if int(fc_output1 / 2) == 4:
                    break
                fc_output1 = int(fc_output1 / 2)
                fc_layers.append(nn.ReLU(True))
                if j % dropout_iterate == 0:
                    fc_layers.append(nn.Dropout(fc_dropout))

        self.fc_layer = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Feeding image through convolutional and pooling layers
        x = self.conv_layer(x)

        # print('x_shape:',x.shape)

        # Flatten
        x = x.view(-1, self.fc_input)  # Flatten the tensor to a 1-D vector

        # output layer of neurons
        x = self.fc_layer(x)

        return x


def main(lr, conv_dropout, fc_dropout, dropout_iterate, conv_layer_output, conv_layer_count, fc_layer_count,
         max_pool_iterate,
         kernel_size, stride_length, train_batch_size, test_batch_size, conv_output_size_iterate,
         model_file_name, num_epochs, patience, model_path):
    with open(model_file_name, "a+") as f:

        num_classes = 4

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Define directories
        dataset_dir = './GeneratedSplitDataset'

        train_dir = os.path.join(dataset_dir, 'train')
        validation_dir = os.path.join(dataset_dir, 'validation')
        test_dir = os.path.join(dataset_dir, 'test')

        # Load the datasets
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        validation_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

        # Create the DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_dataset, batch_size=train_batch_size, shuffle=False,
                                       num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

        print("DataLoaders created from split dataset.")  # DEBUG

        print("Hyperparameters:")
        print(f"Number of epochs: {num_epochs}")
        print(f"Patience value: {patience}")
        print(f"Learning rate: {lr}")
        print(f"Convolutional dropout value: {conv_dropout}")
        print(f"FC layer dropout value: {fc_dropout}")
        print(f"FC dropout iterates after every {dropout_iterate} FC layer")
        print(f"Initial conv layer output channels: {conv_layer_output}")
        print(f"Conv layer number: {conv_layer_count}")
        print(f"FC layer number: {fc_layer_count}")
        print(f"Max pooling iterates after every {max_pool_iterate} conv layer")
        print(f"Kernel size: {kernel_size}")
        print(f"Stride length: {stride_length}")
        print(f"Training & validation batch size: {train_batch_size}")
        print(f"Testing batch size: {test_batch_size}")
        print(f"Conv output channel value doubles after every {conv_output_size_iterate} conv layer")

        model = ConvNeuralNet(conv_dropout, fc_dropout, dropout_iterate, conv_layer_output, conv_layer_count,
                              fc_layer_count, max_pool_iterate,
                              kernel_size, stride_length,
                              conv_output_size_iterate)  # Creating an instance of the CNN

        criterion = nn.CrossEntropyLoss()  # Includes SoftMax, so we do not need a SoftMax activation function at the end of the last fc layer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_steps = len(train_loader)

        best_val_loss = float('inf')
        trigger_times = 0

        min_accuracy = 100
        max_accuracy = 0
        train_loss = 0

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
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs,
                                                                                                i + 1,
                                                                                                total_steps,
                                                                                                loss.item(),
                                                                                                (
                                                                                                        correct / total) * 100))
                    if (correct / total) * 100 < min_accuracy:
                        min_accuracy = (correct / total) * 100
                    if (correct / total) * 100 > max_accuracy:
                        max_accuracy = (correct / total) * 100
                    train_loss = loss.item()
                    f.write(
                        'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(epoch + 1, num_epochs,
                                                                                                i + 1,
                                                                                                total_steps,
                                                                                                loss.item(),
                                                                                                (
                                                                                                        correct / total) * 100))
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

            print('Validation Loss: {:.4f}, Accuracy: {:.4f}%'.format(average_val_loss, accuracy))
            f.write('Validation Loss: {:.4f}, Accuracy: {:.4f}%\n'.format(average_val_loss, accuracy))

            # Early stopping
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                trigger_times = 0

                # Saving best-performing model (based on validation set)
                if os.path.isfile('./' +
                        model_path):  # File exists, will compare best model with current model and will save the better model
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

                    saved_model = ConvNeuralNet(conv_dropout, fc_dropout, dropout_iterate, conv_layer_output, conv_layer_count,
                              fc_layer_count, max_pool_iterate,
                              kernel_size, stride_length,
                              conv_output_size_iterate)  # Model creation as instance of ConvNeuralNet
                    saved_model.load_state_dict(torch.load('./' + model_path))  # Load saved model
                    saved_model_loss = current_saved_model_eval(saved_model, validation_loader,
                                                                criterion)  # Evaluate saved model

                    if average_val_loss < saved_model_loss:  # Compare saved model with current model, save current model as new best model, do nothing otherwise
                        torch.save(model.state_dict(), model_path)
                        print(f"New {model_path} model saved.")

                else:  # File does not exist, first ever model will be saved
                    torch.save(model.state_dict(), model_path)
                    print(f"First {model_path} model saved.")

            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    f.write(f'Early stopping at epoch {epoch + 1}\n')
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
        f.write('Test Accuracy of the model: {} %\n'.format((correct / total) * 100))
    filename = os.path.basename(model_file_name)
    os.rename(model_file_name,
              "./models/{:.4f}__{:.4f}__{:.4f}__{:.4f}__{:.4f}__{}".format((correct / total) * 100, max_accuracy,
                                                                           min_accuracy, train_loss, val_loss,
                                                                           filename))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('pass hyperparameters')
    parser.add_argument('lr', required=True, type=float)
    parser.add_argument('conv_dropout', required=True, type=float)
    parser.add_argument('fc_dropout', required=True, type=float)
    parser.add_argument('dropout_iterate', required=True, type=float)
    parser.add_argument('conv_layer_output', required=True, type=int)
    parser.add_argument('conv_layer_count', required=True, type=int)
    parser.add_argument('fc_layer_count', required=True, type=int)
    parser.add_argument('max_pool_iterate', required=True, type=int)
    parser.add_argument('kernel_size', required=True, type=int)
    parser.add_argument('stride_length', required=True, type=int)
    parser.add_argument('train_batch_size', required=True, type=int)
    parser.add_argument('test_batch_size', required=True, type=int)
    parser.add_argument('conv_output_size_iterate', required=True, type=int)
    parser.add_argument('model_file_name', required=True, type=str)
    parser.add_argument('epoch_number', required=True, type=int)
    parser.add_argument('patience', required=True, type=int)
    parser.add_argument('model_path', required=False, default="best_model.pth", type=str)
    args = parser.parse_args()

    main(args.lr, args.conv_dropout, args.fc_dropout, args.dropout_iterate, args.conv_layer_output,
         args.conv_layer_count, args.fc_layer_count, args.max_pool_iterate,
         args.kernel_size, args.stride_length, args.train_batch_size,
         args.test_batch_size, args.conv_output_size_iterate, args.model_file_name, args.epoch_number, args.patience, args.model_path)
