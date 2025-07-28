# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:23:41 2024

@author: SerbanCaia, Meliimoon
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import KFoldCrossValidationMainCNNModel as main_cnn
import os
from torch.utils.data import Dataset
import numpy as np


# Evaluate dataset with a given model
def evaluate_dataset(dataset_loader, model):
    confusion_matrix = np.zeros((4, 4))
    class_int = 0

    with torch.no_grad():
        for images, labels in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                if class_int != labels[i]:
                    class_int += 1
                confusion_matrix[class_int][predicted[i]] += 1

    return confusion_matrix


# Calculate the sum of the confusion matrix
def calculate_confusion_matrix_sum(confusion_matrix):
    conf_matrix_sum = 0

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            conf_matrix_sum += confusion_matrix[i][j]

    return conf_matrix_sum


# Count true positives for a given class
def count_class_true_positive(confusion_matrix, model_class):
    return confusion_matrix[model_class][model_class]


# Count false positives for a given class
def count_class_false_positive(confusion_matrix, model_class):
    fp = 0

    for j in range(len(confusion_matrix[model_class])):
        if model_class == j:
            continue
        else:
            fp += confusion_matrix[j][model_class]

    return fp


# Count false negatives for a given class
def count_class_false_negative(confusion_matrix, model_class):
    fn = 0

    for j in range(len(confusion_matrix[model_class])):
        if model_class == j:
            continue
        else:
            fn += confusion_matrix[model_class][j]

    return fn


# Count true negatives for a given class
def count_class_true_negative(confusion_matrix, model_class):
    return calculate_confusion_matrix_sum(confusion_matrix) - (
            count_class_true_positive(confusion_matrix, model_class) + count_class_false_positive(
        confusion_matrix, model_class) + count_class_false_negative(confusion_matrix,
        model_class))


# Calculate precision for a given class
def calculate_class_precision(confusion_matrix, model_class):
    return count_class_true_positive(confusion_matrix, model_class) / (
                count_class_true_positive(confusion_matrix, model_class) + count_class_false_positive(confusion_matrix, model_class))


# Calculate recall for a given class
def calculate_class_recall(confusion_matrix, model_class):
    return count_class_true_positive(confusion_matrix, model_class) / (
                count_class_true_positive(confusion_matrix, model_class) + count_class_false_negative(confusion_matrix, model_class))


# Calculate F1 score for a given class
def calculate_class_f1_measure(confusion_matrix, model_class):
    return (2 * calculate_class_precision(confusion_matrix, model_class) * calculate_class_recall(confusion_matrix, model_class)) / (
            calculate_class_precision(confusion_matrix, model_class) + calculate_class_recall(confusion_matrix, model_class))


# Calculate overall accuracy
def calculate_accuracy(confusion_matrix):
    tp = 0

    for model_class in range(len(confusion_matrix)):
        tp += count_class_true_positive(confusion_matrix, model_class)

    return tp / calculate_confusion_matrix_sum(confusion_matrix)


def main(fold, test_dataloader):

    # Declare list of class names and performance metrics table
    classes_list = ['Angry', 'Focused', 'Happy', 'Neutral']
    performance_metrics_tabular = [
        ["Fold", "Macro-Precision", "Macro-Recall", "Macro-F1", "Micro-Precision", "Micro-Recall", "Micro-F1",
         "Accuracy"], []]

    # Declare confusion matrix properties
    confusion_matrix_tabular = [[], [], [], [], []]

    chosen_model = f'./main_kfold_best_model_fold{fold}.pth'
    model = main_cnn.ConvNeuralNet()  # Create instance of ConvNeuralNet
    model.load_state_dict(torch.load(chosen_model))  # Load the model

    model.eval()

    confusion_matrix = evaluate_dataset(test_dataloader, model)

    # Calculate performance metrics
    accuracy = calculate_accuracy(confusion_matrix)

    macro_precision = 0
    micro_precision_numerator = 0
    micro_precision_denominator = 0

    macro_recall = 0
    micro_recall_numerator = 0
    micro_recall_denominator = 0

    macro_f1 = 0
    micro_f1_numerator = 0
    micro_f1_denominator = 0

    for class_int in range(len(classes_list)):
        # Add the current class' precision, recall, and f1-measure to the macro-precision, macro-recall,
        # and macro-f1-measure variables respectively
        macro_precision += calculate_class_precision(confusion_matrix, class_int)
        macro_recall += calculate_class_recall(confusion_matrix, class_int)
        macro_f1 += calculate_class_f1_measure(confusion_matrix, class_int)

        # Add the current class' true positive score to the numerator variables
        micro_precision_numerator += count_class_true_positive(confusion_matrix, class_int)
        micro_recall_numerator += count_class_true_positive(confusion_matrix, class_int)
        micro_f1_numerator += count_class_true_positive(confusion_matrix, class_int)

        # Add the current class' tp+fp, tp+fn, and tp+(1/2)(fp + fn) to the respective denominator variable
        micro_precision_denominator += count_class_true_positive(confusion_matrix, class_int) + count_class_false_positive(confusion_matrix, class_int)
        micro_recall_denominator += count_class_true_positive(confusion_matrix, class_int) + count_class_false_negative(confusion_matrix, class_int)
        micro_f1_denominator += count_class_true_positive(confusion_matrix, class_int) + (1 / 2) * (
                    count_class_false_positive(confusion_matrix, class_int) + count_class_false_negative(confusion_matrix, class_int))

    # Compute the model's macro metrics
    macro_precision /= len(classes_list)
    macro_recall /= len(classes_list)
    macro_f1 /= len(classes_list)

    # Compute the model's micro metrics
    micro_precision = micro_precision_numerator / micro_precision_denominator
    micro_recall = micro_recall_numerator / micro_recall_denominator
    micro_f1 = micro_f1_numerator / micro_f1_denominator

    classes_row = []

    # Fill confusion matrix table with headers
    for class_int in range(len(classes_list) + 1):
        if class_int == 0:
            classes_row.append("Class")
        else:
            classes_row.append(classes_list[class_int - 1])

    confusion_matrix_tabular[0] = classes_row

    # Fill confusion matrix table with performance metrics
    for class_int in range(1, len(confusion_matrix_tabular)):
        for column in range(len(confusion_matrix_tabular[0])):
            if column == 0:
                confusion_matrix_tabular[class_int].append(classes_list[class_int - 1][column - 1])
            else:
                confusion_matrix_tabular[class_int].append(str(confusion_matrix[class_int - 1][column - 1]))

    # Display the current model's confusion matrix and performance metrics
    print(f"\n{chosen_model}'s Confusion Matrix:")
    for class_int in range(len(classes_list)):
        print(f'{confusion_matrix[class_int]}')

    performance_metrics_tabular[1] = [f"{fold}", f'{macro_precision * 100:.4f}%',
                                                         f'{macro_recall * 100:.4f}%', f'{macro_f1 * 100:.4f}%',
                                                         f'{micro_precision * 100:.4f}%', f'{micro_recall * 100:.4f}%',
                                                         f'{micro_f1 * 100:.4f}%', f'{accuracy * 100:.4f}%']
    print(f"\nFold {fold}'s Performance Metrics:\n{performance_metrics_tabular[0]}\n{performance_metrics_tabular[1]}\n")

    return performance_metrics_tabular[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('pass arguments')
    parser.add_argument('fold', required=True, type=int)
    parser.add_argument('test_dataloader', required=True, type=DataLoader)
    args = parser.parse_args()

    main(args.fold, args.test_dataloader)
