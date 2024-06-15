# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:08:46 2024

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

# Optionally define initial hardcoded hyperparameter values
"""
lr = 50
conv_dropout = 2.5
fc_dropout = 5
dropout_iterate = 1
conv_layer_output = 5
conv_layer_count = 5
fc_layer_count = 2
max_pool_iterate = 2
kernel_size = 3
stride_length = 1
train_batch_size = 6
conv_output_size_iterate = 2
epoch_num = 25
patience = 6
"""

# Iterate over different hyperparameter combinations
for train_batch_size in range(6, 9):
    for stride_length in range(1, 6):
        for kernel_size in range(3, 11, 2):
            for conv_dropout in range(-1, 6, 1):
                if conv_dropout == -1:
                    conv_dropout = 2.5
                for fc_dropout in range(5, 0, -1):
                    for conv_output_size_iterate in range(1, 6):
                        for max_pool_iterate in range(1, 11):
                            for dropout_iterate in range(1, 6):
                                for conv_layer_count in range(1, 11):
                                    for fc_layer_count in range(1, 6):
                                        for conv_layer_output in range(3, 10):
                                            for lr in range(0, 275, 25):
                                                if lr == 0:
                                                    lr = 10
                                                for epoch_num in range(10, 35, 5):
                                                    for patience in range(3, 7):
                                                        for i in range(1, 6):
                                                            # Create and write the configuration for each model
                                                            with open(f"./models/model{nextModelCount}.txt", 'a+') as f:
                                                                nextModelCount += 1
                                                                with open("./model_count_config.txt", 'w') as configFile:
                                                                    configFile.write(str(nextModelCount))
                                                                f.write(
                                                                    f"Training & validation batch size: {2 ** train_batch_size}\n" +
                                                                    f"Testing batch size: {2 ** (train_batch_size + 1)}\n" +
                                                                    f"Stride length: {stride_length}\n" +
                                                                    f"Kernel size: {kernel_size}\n" +
                                                                    f"Convolutional dropout value: {conv_dropout / 10}\n" +
                                                                    f"FC layer dropout value: {fc_dropout / 10}\n" +
                                                                    f"Max pooling iterates after every {max_pool_iterate} conv layer\n" +
                                                                    f"Conv output channel value doubles after every {conv_output_size_iterate} conv layer\n" +
                                                                    f"FC dropout iterates after every {dropout_iterate} FC layer\n" +
                                                                    f"Conv layer number: {conv_layer_count}\n" +
                                                                    f"FC layer number: {fc_layer_count}\n" +
                                                                    f"Initial conv layer output channels: {2 ** conv_layer_output}\n" +
                                                                    f"Learning rate: {str(lr / 100000)}\n" +
                                                                    f"Number of epochs: {epoch_num}\n" +
                                                                    f"Patience value: {patience}\n" +
                                                                    f"Try #{i}\n\n")
                                                            # Call the main function of cnnmodel_automated with current hyperparameters
                                                            cnnmodel_automated.main(lr / 100000, conv_dropout / 10,
                                                                                    fc_dropout / 10,
                                                                                    dropout_iterate, 2 ** conv_layer_output,
                                                                                    conv_layer_count, fc_layer_count,
                                                                                    max_pool_iterate,
                                                                                    kernel_size, stride_length,
                                                                                    2 ** train_batch_size,
                                                                                    2 ** (train_batch_size + 1),
                                                                                    conv_output_size_iterate,
                                                                                    f"./models/model{nextModelCount - 1}.txt",
                                                                                    epoch_num, patience, 'main_best_model.pth')
