import pandas as pd
import numpy as np

header_names = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]


class DataCleaning:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataframe = self.load()

    def load(self):
        return pd.read_csv(self.filepath, header=None, names=header_names)

    def read_head(self):
        return self.dataframe.head()

    def classifier_update(self):
        self.dataframe["Income"] = self.dataframe["Income"].map({"<=50K": -1, ">50K": 1})
