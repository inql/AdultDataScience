from adult.database import Database
from adult import stats, fill_missing, classification, association, clustering

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
print(test.all_values.head())
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
print('\nMap income to {-1,1}\n')
test.map_income()
print('\nDisplay NaN values in this database (marked as ?)\n')
print(test.get_na())
print('\nSTATS\n')
stats.count_values(test)
stats.count_column_values(test)
stats.get_math_stats(test)

print('\n---Filling missing data based on classification algorithms---\n')
fill_missing.fill_question_marks_based_on_predicting(test, 'WorkClass', 0.05)
fill_missing.fill_question_marks_based_on_predicting(test, 'Occupation', 0.05)
fill_missing.fill_question_marks_based_on_predicting(test, 'NativeCountry', 0.05)
print('Check the result - now NaN values should not be available')
test.get_na()
print('Reset the categories')
test.reset_categories()

print('\n-- Classification -- \n')
test.get_classes()
test.get_all_inputs()
test.split_data_to_test()

c_model = classification.ClassificationModel()
c_model.perform_all(test)

print(c_model.acc)
print(c_model.conf_matrix)


print('\n-- Apriori algorirthm -- \n')
test.get_apriori()
test.get_onehot_values()
association.perform_apriori(test)

test.get_classes()
test.get_all_inputs()
test.split_data_to_test()
test.get_normalized_test_input()
labels = clustering.perform_dbscan(test)