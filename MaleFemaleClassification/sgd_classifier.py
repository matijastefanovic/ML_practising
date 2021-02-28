import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score


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

    classifier.fit(X_train_scaled, y_train.values.ravel())

    y_found = classifier.predict(X_train_scaled)
    matrix = confusion_matrix(y_true=y_train, y_pred=y_found)
    print(matrix)
    print('sgd_classfier w/o optimizations')
    print(roc_auc_score(y_train.values.ravel(), classifier.decision_function(X_train_scaled)))
    y_train_pred_cross_val = cross_val_predict(classifier, X_train_scaled, y_train.values.ravel(), cv=1200, n_jobs=-1, verbose=3)
    matrix = confusion_matrix(y_true=y_train, y_pred=y_train_pred_cross_val)
    print('sgd_classfier w/o optimizations but from cross val predict')
    print(matrix)
    # here I want to get thresholds mapped

