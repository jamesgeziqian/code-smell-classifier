import os
import time

import numpy as np
import pandas as pd

from scipy.io import arff
from sklearn import ensemble, naive_bayes, svm, tree
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV


def cast_byte_to_bool(y):
    return [yy == b'true' for yy in y]


def get_data_np(file_name, y_column_name):
    data_raw = arff.loadarff(os.path.join('data', file_name))
    data_df = pd.DataFrame(data_raw[0]).dropna(axis=0)
    data_np = data_df.drop(y_column_name, axis=1).to_numpy()
    labels_np = cast_byte_to_bool(data_df.loc[:, y_column_name].to_numpy())
    return data_np, labels_np


def print_grid_search_results(clf: GridSearchCV, X_test, y_test):
    print(f'Best val score: {clf.best_score_}')
    print(f'Best params: {clf.best_params_}')
    print(
        f'Average training time: {np.mean(clf.cv_results_["mean_fit_time"])}')
    print(f'Test accuracy with best params: {clf.score(X_test, y_test)}')
    print(
        f'Test f1 score with best params: {f1_score(y_test, clf.predict(X_test))}')
    print(f'Test recall: {recall_score(y_test, clf.predict(X_test))}')
    print(f'Test precision: {precision_score(y_test, clf.predict(X_test))}')


def svc_grid_search(X_train, X_test, y_train, y_test, n_jobs=1, **params):
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid=params or {
        'C': [1, 10],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': list(range(1, 5)),
    }, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    print_grid_search_results(clf, X_test, y_test)
    return clf.best_estimator_


def train_test_naive_bayes(X_train, X_test, y_train, y_test):
    nb = naive_bayes.GaussianNB()
    start = time.time()
    nb.fit(X_train, y_train)
    end = time.time()
    print(f'Training time: {end - start}')
    print(f'Test accuracy: {nb.score(X_test, y_test)}')
    print(f'Test f1 score: {f1_score(y_test, nb.predict(X_test))}')
    print(f'Test recall: {recall_score(y_test, nb.predict(X_test))}')
    print(f'Test precision: {precision_score(y_test, nb.predict(X_test))}')
    return nb


def decision_tree_grid_search(X_train, X_test, y_train, y_test, random_state=None, n_jobs=1, **params):
    dt = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dt, param_grid=params or {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_features': ['sqrt', 'log2', None],
        'random_state': [random_state],
    }, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    print_grid_search_results(clf, X_test, y_test)
    return clf.best_estimator_


def random_forest_grid_search(X_train, X_test, y_train, y_test, random_state=None, n_jobs=1, **params):
    rf = ensemble.RandomForestClassifier()
    clf = GridSearchCV(rf, param_grid=params or {
        'n_estimators': list(range(20, 200, 20)),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'random_state': [random_state],
    }, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    print_grid_search_results(clf, X_test, y_test)
    return clf.best_estimator_


def train_models(data, labels, test_size=0.2, random_state=None, n_jobs=1):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state)
    print('==== Support Vector Classifier ====')
    svc_grid_search(X_train, X_test, y_train, y_test, n_jobs=n_jobs)

    print('==== Gaussian Naive Bayes Classifier ====')
    train_test_naive_bayes(X_train, X_test, y_train, y_test)

    print('==== Decision Tree ====')
    decision_tree_grid_search(X_train, X_test, y_train,
                              y_test, random_state=random_state, n_jobs=n_jobs)

    print('==== Random Forest ====')
    random_forest_grid_search(X_train, X_test, y_train,
                              y_test, random_state=random_state, n_jobs=n_jobs)


def main():
    dc_data, dc_labels = get_data_np('data-class.arff', 'is_data_class')
    fe_data, fe_labels = get_data_np('feature-envy.arff', 'is_feature_envy')
    gc_data, gc_labels = get_data_np('god-class.arff', 'is_god_class')
    lm_data, lm_labels = get_data_np('long-method.arff', 'is_long_method')

    n_jobs = -1
    print('=== Data class ===')
    train_models(dc_data, dc_labels, n_jobs=n_jobs, random_state=1)
    print('\n=== Feature envy ===')
    train_models(fe_data, fe_labels, n_jobs=n_jobs, random_state=1)
    print('\n=== God class ===')
    train_models(gc_data, gc_labels, n_jobs=n_jobs, random_state=1)
    print('\n=== Long method ===')
    train_models(lm_data, lm_labels, n_jobs=n_jobs, random_state=1)


if __name__ == '__main__':
    main()
