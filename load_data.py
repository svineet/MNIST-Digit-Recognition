#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

from pathlib import Path

def load_test_data(csv_path):
    data = pd.read_csv(csv_path)
    return data

def load_data(csv_path, number_of_validation_points):
    data = pd.read_csv(csv_path)
    data_train = data[:-number_of_validation_points]
    data_val = data[-number_of_validation_points:]

    X_train, y_train = data_train.drop("label", axis=1), data_train['label']
    X_val, y_val = data_val.drop("label", axis=1), data_val['label']

    return X_train, y_train, X_val, y_val

def main():
    load_data((Path('.') / 'train.csv').absolute())

if __name__ == '__main__':
    main()
