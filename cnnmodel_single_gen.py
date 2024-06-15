# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:14:32 2024

@author: SerbanCaia
"""
import os.path
import cnnmodel_automated

# Check if the model count configuration file exists; if not, create it and initialize with '0'
if not os.path.exists("./model_count_config.txt"):
    with open("./model_count_config.txt", 'w') as configFile:
        configFile.write('0')

# Read the next model count from the configuration file
with open("./model_count_config.txt", 'r+') as configFile:
    nextModelCount = int(configFile.readline())


# Prompt user to choose which model to save
model_path = ''
while True:
    option = input("Which model would you like to save? Main model (0), or a variant (1)? (type only one of the numbers in parentheses) ")
    if option == "0":
        model_path = 'best_model.pth'
    elif option == "1":
        while True:
            if os.path.exists(f'./variant{option}.pth'):
                option = str(int(option) + 1)
                continue
            else:
                model_path = f'variant{option}.pth'
                print(f"Creating Variant {option}\n")
                break
        break
    else:
        print("Invalid input. Please choose another one\n")

# Prompt user to input hyperparameter values (commented hard-coded hyperparameters below each input method for testing)
lr = float(input("Which learning rate value would you like to use? "))
#lr = 0.0005
conv_dropout = float(input("Which convolutional dropout value would you like to use? "))
#conv_dropout = 0.25
fc_dropout = float(input("Which fully connected dropout value would you like to use? "))
#fc_dropout = 0.5
dropout_iterate = int(input("After how many iterations of fully connected layers would you like the fc dropout to be applied? "))
#dropout_iterate = 1
conv_layer_output = int(input("How many output channels would you like the first convolutional layer to have? "))
#conv_layer_output = 32
conv_layer_count = int(input("How many convolutional layers would you like to have? "))
#conv_layer_count = 5
fc_layer_count = int(input("How many fully connected layers would you like to have? "))
#fc_layer_count = 2
max_pool_iterate = int(input("After how many iterations of convolutional layers would you like a max pooling to be applied? "))
#max_pool_iterate = 2
kernel_size = int(input("What would you like your kernel height/width value to be? "))
#kernel_size = 3
stride_length = int(input("What would you like your stride length to be? "))
#stride_length = 1
train_batch_size = int(input("What size training & validation batch would you like to have? "))
#train_batch_size = 64
conv_output_size_iterate = int(input("After how many iterations of convolutional layers would you like your amount of output channels to double? "))
#conv_output_size_iterate = 2
epoch_num = int(input("How many epochs would you like to have? "))
#epoch_num = 25
patience = int(input("How many epochs would you like the patience value to be? "))
#patience = 6

print(f"\nModel will be run 5 times\n")

# Run the model 5 times with the specified hyperparameters
for i in range(1, 6):
    # Create and write the configuration for each model
    with open(f"./models/model{nextModelCount}.txt", 'a+') as f:
        nextModelCount += 1
        with open("./model_count_config.txt", 'w') as configFile:
            configFile.write(str(nextModelCount))
        f.write(
            f"Contested for model path: {model_path}\n" +
            f"Training & validation batch size: {train_batch_size}\n" +
            f"Testing batch size: {2 * train_batch_size}\n" +
            f"Stride length: {stride_length}\n" +
            f"Kernel size: {kernel_size}\n" +
            f"Convolutional dropout value: {conv_dropout}\n" +
            f"FC layer dropout value: {fc_dropout}\n" +
            f"Max pooling iterates after every {max_pool_iterate} conv layer\n" +
            f"Conv output channel value doubles after every {conv_output_size_iterate} conv layer\n" +
            f"FC dropout iterates after every {dropout_iterate} FC layer\n" +
            f"Conv layer number: {conv_layer_count}\n" +
            f"FC layer number: {fc_layer_count}\n" +
            f"Initial conv layer output channels: {conv_layer_output}\n" +
            f"Learning rate: {str(lr)}\n" +
            f"Number of epochs: {epoch_num}\n" +
            f"Patience value: {patience}\n" +
            f"Try #{i}\n\n")
    # Call the main function of cnnmodel_automated with current hyperparameters
    cnnmodel_automated.main(lr, conv_dropout,
                            fc_dropout,
                            dropout_iterate, conv_layer_output,
                            conv_layer_count, fc_layer_count,
                            max_pool_iterate,
                            kernel_size, stride_length,
                            train_batch_size,
                            2 * train_batch_size,
                            conv_output_size_iterate,
                            f"./models/model{nextModelCount - 1}.txt",
                            epoch_num, patience, model_path)
