#!/usr/bin/python3

import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def get_accuracies_vs_features(directory):
    accuracies = []
    for feature_count in range(2, 58):
        # read from file
        data = pd.read_csv(directory + str(feature_count) + "_features.csv", header=None, index_col=feature_count)

        # shuffle data and split into training, testing, and target values for each
        training_data, testing_data, training_targets, testing_targets = sklearn.model_selection \
            .train_test_split(data, data.index.values, test_size=0.5)

        # get a scaler object to scale both the training and testing data
        scaler = sklearn.preprocessing.StandardScaler().fit(training_data)

        # scale the training and testing data
        training_data = pd.DataFrame(scaler.transform(training_data), index=training_targets)
        testing_data = pd.DataFrame(scaler.transform(testing_data), index=testing_targets)

        # create and train SVM
        classifier = SVC(kernel="linear", probability=True,)
        classifier.fit(training_data, training_targets)

        # get predictions
        accuracy = classifier.score(testing_data, testing_targets)
        accuracies.append(accuracy)
    return accuracies


if __name__ == "__main__":
    # get the accuracies for features selected by weight
    weighted_accuracies = get_accuracies_vs_features("highest_weight_feature_data/")
    # get the accuracies for features selected randomly
    random_accuracies = get_accuracies_vs_features("random_feature_data/")
    # plot the two
    plt.title("Top features vs Random features")
    plt.plot(list(range(2, 58)), weighted_accuracies, 'blue', label="Best features")
    plt.plot(list(range(2, 58)), random_accuracies, 'red', label="Random features")
    plt.legend(loc='lower right')
    plt.xlim([0, 60])
    plt.ylim([0.6, 1])
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Features")
    plt.show()
    pass
