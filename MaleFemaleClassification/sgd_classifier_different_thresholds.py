import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
import numpy as np

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    # plt.axis([-50000, 50000, 0, 1])             # Not shown

if __name__ == '__main__':
    pd.set_option("mode.chained_assignment", None)
    path = "data/gender.csv"
    df = pd.read_csv(path)

    stratisfied_split = StratifiedShuffleSplit(test_size=0.2, random_state=42)

    for train_index, test_index in stratisfied_split.split(df, df['Gender'], groups=df['Gender']):
        strat_test_set: pd.DataFrame = df.loc[test_index]
        strat_train_set: pd.DataFrame = df.loc[train_index]


    y_train = strat_train_set[['Gender']]
    X_train = strat_train_set.drop('Gender', axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    y_train['Gender'] =  y_train.loc[:, 'Gender'] == 'Male'

    classifier = SGDClassifier(loss='log', max_iter=10000, n_jobs=-1)

    y_scores = cross_val_predict(classifier, X_train_scaled, y_train.values.ravel(), cv=120, n_jobs=-1,
                                               verbose=3, method='decision_function')

    precisions, recalls, thresholds = precision_recall_curve(y_true=y_train, probas_pred=y_scores)
    recall_90_precision = recalls[np.argmax(precisions >= 0.95)]
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.95)]

    # plt.figure()
    # plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
    # plt.plot(thresholds, recalls[:-1], "g-", label='Recall')
    # plt.legend()
    # plt.grid()
    # plt.show()
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.95], "r:")  # Not shown
    plt.plot([-10, threshold_90_precision], [0.95, 0.95], "r:")  # Not shown
    plt.plot([-10, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")  # Not shown
    plt.plot([threshold_90_precision], [0.95], "ro")  # Not shown
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")  # Not shown