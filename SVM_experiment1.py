#!/usr/bin/python3

import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

# read from file
data = pd.read_csv("./spambase/spambase.data", header=None, index_col=57)

# shuffle data and split into training, testing, and target values for each
training_data, testing_data, training_targets, testing_targets = sklearn.model_selection\
    .train_test_split(data, data.index.values, test_size=0.5)

# get a scaler object to scale both the training and testing data
scaler = sklearn.preprocessing.StandardScaler().fit(training_data)

# scale the training and testing data
training_data = pd.DataFrame(scaler.transform(training_data), index=training_targets)
testing_data = pd.DataFrame(scaler.transform(testing_data), index=testing_targets)

# create and train SVM
classifier = SVC(kernel="linear", probability=True,)
classifier.fit(training_data, training_data.index.values)

# get probabilities of correct classification
decisions = classifier.decision_function(testing_data)

# get accuracy of classification
score = classifier.score(testing_data, testing_targets)
print("accuracy " + str(score))
# get precision of classicfication
precision = average_precision_score(testing_targets, decisions)
print("precision " + str(precision))

# compare predicted values with actual values
false_positive_rate, true_positive_rate, thresholds = roc_curve(testing_targets, decisions)
roc_auc = auc(false_positive_rate, true_positive_rate)


plt.title("spambase ROC")
plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'm--')
plt.xlim([0, 1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
pass
