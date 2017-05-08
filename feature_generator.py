#!/usr/bin/python3

import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import os
import random

def rank_features(data: np.array):
    """ This will return a list of rankings for the sizes of the columns"""
    # holds the sums of each column
    column_sums = []
    # holds the rank of each column
    colum_rank = []

    # sum each column and put the sum into the list
    for x in range(data.shape[1]):
        column_sum = 0
        for y in range(data.shape[0]):
            column_sum += data[y][x]
        column_sums.append(column_sum)
        # add a place holder
        colum_rank.append(0)
    # rank the columns
    for i in range(len(column_sums)):
        # find the biggest one
        max_index = column_sums.index(max(column_sums))
        # remove it while preserving the shape of the list
        column_sums[max_index] = - 1
        # assign that rank to the placeholder
        colum_rank[max_index] = i
    return colum_rank


def write_features(file, data, targets, ranks, size):
    """will write features and targets to file"""
    for instance in range(data.shape[0]):
        features = ""
        for feature_index in range(len(ranks)):
            if ranks[feature_index] < size:
                features += str(data[instance][feature_index]) + ","
        line = features + str(targets[instance])
        file.write(line + "\n")

if __name__ == "__main__":
    # read from file
    data = pd.read_csv("./spambase/spambase.data", header=None, index_col=57)

    # turn everything into np.arrays because DataFrames are weird
    targets = np.array(data.index.values)
    data = np.array(data)

    weighted_ranks = rank_features(data)
    directory = "./highest_weight_feature_data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # generate data with best features
    for num_features in range(2, 58):
        data_file = open(directory + str(num_features) + "_features.csv", 'w')
        write_features(data_file, data, targets, weighted_ranks, num_features)
        data_file.close()

    random_ranks = list(range(57))
    random.shuffle(random_ranks)
    directory = "./random_feature_data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # generate data with random features
    for num_features in range(2, 58):
        data_file = open(directory + str(num_features) + "_features.csv", 'w')
        write_features(data_file, data, targets, random_ranks, num_features)
        data_file.close()
