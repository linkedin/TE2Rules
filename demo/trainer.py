"""
This file has class to train a tree ensemble model used in TE2Rules experiments in the paper.
"""
import numpy as np
import pandas as pd
from sklearn import metrics


class Trainer:
    def __init__(self, training_data_loc, testing_data_loc, scikit_model):
        self.set_seed()
        self.read_data(training_data_loc, testing_data_loc)
        self.train_model(scikit_model)

    def set_seed(self, seed=123):
        np.random.seed(123)

    def read_data(self, training_data_loc, testing_data_loc):
        data_train = pd.read_csv(training_data_loc)
        data_test = pd.read_csv(testing_data_loc)
        cols = list(data_train.columns)

        data_train = data_train.to_numpy()
        data_test = data_test.to_numpy()

        self.x_train = data_train[:, :-1]
        self.y_train = data_train[:, -1]

        self.x_test = data_test[:, :-1]
        self.y_test = data_test[:, -1]

        self.feature_names = cols[:-1]
        self.label_name = cols[-1]

    def train_model(self, scikit_model):
        self.model = scikit_model
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        accuracy = self.model.score(self.x_test, self.y_test)

        y_pred = self.model.predict_proba(self.x_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_pred)
        auc = metrics.auc(fpr, tpr)

        return accuracy, auc
