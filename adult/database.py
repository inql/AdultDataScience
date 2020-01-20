import pandas as pd


class Database:
    def __init__(self):
        self.values = {
            'all_values': None,
            'apriori_values': None,
            'columns': [
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
            ],
            'missing_values': [''],
            'all_inputs': None,
            'all_classes': None,
            'train_inputs': None,
            'test_inputs': None,
            'train_classes': None,
            'test_classes': None,
            'onehot': None
        }

    def get_data(self):
        self.values['all_values'] = pd.read_csv(
            'uci-adult/data/adult_data.csv',
            na_values=self.values['missing_values'], names=self.values['columns'])

    def get_info(self):
        return self.values['all_values'].info()

    def set_all_categorical_columns(self):
        for col in set(self.values['all_values'].columns) - set(self.values['all_values'].describe().columns):
            self.values['all_values'][col] = self.values['all_values'][col].astype('category')

    def fix_income(self):
        self.values['all_values'].Income = self.values['all_values'].Income.replace("50k","<=50K")

    def map_income(self):
        self.values['all_values'].Income = self.values['all_values'].Income.map({"<=50K": -1, ">50K": 1})

    def get_classes(self):
        self.values['all_classes'] = self.values['all_values'].Income.values

    def income_uniques(self):
        return self.values['all_values'].Income.unique()

    def income_count(self):
        return self.values['all_values'].Income.value_counts()

    def get_na(self):
        return self.values['all_values'].isna().sum()
