# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:25:51 2024

@author: SerbanCaia
"""
import MainCNNModel_callable as main_cnn
import Variant1Model_callable as variant1_cnn
import Variant2Model_callable as variant2_cnn

# Choose which model to save
while True:
    option = input("Which model would you like to save? Main model (0), variant 1 (1), or variant 2 (2)? (type only one of the numbers in parentheses) ")
    if option == "0":
        print(f"\nModel will be run 5 times\n")
        for i in range(1, 6):
            main_cnn.main()
        break
    elif option == "1":
        print(f"\nModel will be run 5 times\n")
        for i in range(1, 6):
            variant1_cnn.main()
        break
    elif option == "2":
        print(f"\nModel will be run 5 times\n")
        for i in range(1, 6):
            variant2_cnn.main()
        break
    else:
        print("Invalid input. Please choose another one\n")
