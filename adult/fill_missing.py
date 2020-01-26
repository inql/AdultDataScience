import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree


def get_one_hot_from_categories(dataframe):
    data_cols = dataframe.select_dtypes('category').columns

    data_cols_dropped = dataframe.drop(columns=data_cols, axis=1)
    one_hot = pd.get_dummies(dataframe[data_cols])

    return pd.concat([data_cols_dropped, one_hot], axis=1, join='inner')


def fill_question_marks_based_on_predicting(database, column, test_percentage):
    print('Column refactored - ' + str(column))
    test_data = database.all_values[(database.all_values[column].values == '?')].copy()
    test_label = test_data[column]

    train_data = database.all_values[(database.all_values[column].values != '?')].copy().sample(
        frac=0.1)
    train_label = train_data[column]

    test_data.drop(columns=[column], inplace=True)
    train_data.drop(columns=[column], inplace=True)

    train_data = get_one_hot_from_categories(train_data)
    test_data = get_one_hot_from_categories(test_data)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_label)
    clf_pred = clf.predict(test_data)

    r_forest = RandomForestClassifier(n_estimators=10)
    r_forest.fit(train_data, train_label)
    r_forest_pred = r_forest.predict(test_data)

    majority_class = database.all_values[column].value_counts().index[0]

    pred_df = pd.DataFrame({'RFor': r_forest_pred, 'DTree': clf_pred})
    overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts()[0] > 1 else majority_class,
                                 axis=1)
    print(overall_pred)
    database.all_values.loc[
        (database.all_values[column].values == '?'), column] = overall_pred.values
    print(database.all_values[column].value_counts())
    print(database.all_values[column].unique())
