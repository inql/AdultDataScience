from adult.database import Database
from adult import stats, fill_missing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

import pandas as pd

print('Uci-Adult database analysis.\nAuthor: Dawid Bińkuś 246793\n\n')
print('Load and display head of database:\n')
test = Database()
test.get_data()
print(test.values['all_values'].head())
test.set_all_categorical_columns()
print(test.get_info())
print('\nShow unique values of Income column:\n')
print(test.income_uniques())
print('\nCount unique values of Income column:\n')
print(test.income_count())
print('\nSeems like one of the value is invalid - let\'s fix it by assigning it assuming 50k == <=50k')
test.fix_income()
print('\nDisplay unique values of Income column:\n')
print(test.income_count())
print('\nDisplay NaN values in this database (marked as ?)\n')
print(test.get_na())
print('\nSTATS\n')
stats.count_values(test)
stats.count_column_values(test)
stats.get_math_stats(test)

print('\n---Filling missing data based on classification algorithms---\n')
test_data = test.values['all_values'][(test.values['all_values'].WorkClass.values == '?')].copy()
test_label = test_data.WorkClass

train_data = test.values['all_values'][(test.values['all_values'].WorkClass.values != '?')].copy()
train_label = train_data.WorkClass

test_data.drop(columns = ['WorkClass'], inplace = True)
train_data.drop(columns = ['WorkClass'], inplace = True)

train_data = fill_missing.get_one_hat_from_categories(train_data)
test_data = fill_missing.get_one_hat_from_categories(test_data)

log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(train_data[0:1000], train_label[0:1000])
log_reg_pred = log_reg.predict(test_data)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data[0:1000], train_label[0:1000])
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data[0:1000], train_label[0:1000])
r_forest_pred = r_forest.predict(test_data)

majority_class = test.values['all_values'].WorkClass.value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts()[0] > 1 else majority_class, axis = 1)
print(overall_pred)
test.values['all_values'].loc[(test.values['all_values'].WorkClass.values == '?'),'WorkClass'] = overall_pred.values
print(test.values['all_values'].WorkClass.value_counts())
print(test.values['all_values'].WorkClass.unique())