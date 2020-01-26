import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def perform_apriori(data):
    items = apriori(data.one_hot_values, min_support=0.4, use_colnames=True, verbose=0)
    rules = association_rules(items, metric='confidence', min_threshold=0.6)
    print(rules)