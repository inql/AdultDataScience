import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class Database:
    def __init__(self):
        self.all_values = None
        self.apriori_values = None
        self.columns = [
            "Age",
            "WorkClass",
            "fnlwgt",
            "Education",
            "EducationNum",
            "MaritalStatus",
            "Occupation",
            "Relationship",
            "Race",
            "Gender",
            "CapitalGain",
            "CapitalLoss",
            "HoursPerWeek",
            "NativeCountry",
            "Income"
        ]
        self.all_inputs = None
        self.all_classes = None
        self.train_inputs = None
        self.test_inputs = None
        self.train_classes = None
        self.test_classes = None
        self.train_normalized = None
        self.one_hot_values = None

    def get_data(self):
        self.all_values = pd.read_csv(
            'uci-adult/data/adult_data.csv',
            names=self.columns).drop(columns=['fnlwgt'])
        self.columns.remove('fnlwgt')

    def get_apriori(self):
        self.apriori_values = self.all_values.copy()
        self.apriori_values = self.apriori_values.drop(
            columns=['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek'])

    def get_info(self):
        return self.all_values.info()

    def set_all_categorical_columns(self):
        for col in set(self.all_values.columns) - set(self.all_values.describe().columns):
            self.all_values[col] = self.all_values[col].astype('category')

    def fix_income(self):
        self.all_values.Income = self.all_values.Income.replace("50k", "<=50K")

    def map_income(self):
        self.all_values.Income = self.all_values.Income.map({"<=50K": 0, ">50K": 1})

    def fix_gender(self):
        self.all_values.Gender = self.all_values.Gender.replace("male", "Male")

    def fix_race(self):
        self.all_values.Race = self.all_values.Race.replace("white", "White")

    def fix_occupation(self):
        self.all_values.Occupation = self.all_values.Occupation.replace("adm_clerical", "Adm-clerical")

    def get_classes(self):
        self.all_classes = self.all_values.Income.values

    def get_all_inputs(self):
        self.all_inputs = self.all_values.copy()
        self.all_inputs.drop(columns=['Income'], inplace=True)

    def split_data_to_test(self):
        adult_cat_1hot = pd.get_dummies(self.all_inputs.select_dtypes('category'))
        adult_non_cat = self.all_inputs.select_dtypes(exclude='category')
        adult_data_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')
        (self.train_inputs, self.test_inputs,
         self.train_classes, self.test_classes) = \
            train_test_split(adult_data_1hot, self.all_classes, train_size=0.25)

    def income_uniques(self):
        return self.all_values.Income.unique()

    def income_count(self):
        return self.all_values.Income.value_counts()

    def get_na(self):
        for i, j in zip(self.columns, (self.all_values.values.astype(str) == '?').sum(axis=0)):
            if j > 0:
                print(str(i) + ': ' + str(j) + ' records')

    def get_onehot_values(self):
        self.one_hot_values = pd.get_dummies(self.apriori_values)

    def reset_categories(self):
        self.all_values['WorkClass'] = self.all_values['WorkClass'].cat.remove_categories('?')
        self.all_values['Occupation'] = self.all_values['Occupation'].cat.remove_categories('?')
        self.all_values['NativeCountry'] = self.all_values['NativeCountry'].cat.remove_categories('?')

    def get_normalized_test_input(self):
        normalized_test_input = normalize(pd.get_dummies(self.train_inputs))
        pca = PCA(n_components=2)
        normalized_df = pd.DataFrame(pca.fit_transform(normalized_test_input))
        normalized_df.columns = ['P1', 'P2']
        self.train_normalized = normalized_df
