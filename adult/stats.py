def count_values(data):
    print('DATA COUNT:\n')
    print(data.values['all_values'].count())


def count_column_values(data):
    print('Values per column:\n')
    for column in data.values['columns']:
        print(data.values['all_values'][column].value_counts(), 'n')


def get_math_stats(data):
    print('\nStatistics from num data:\n')
    for column in set(data.values['all_values'].describe().columns):
        print('\n' + column, '\n')
        col_data = data.values['all_values'][column]
        print('Mean: ' + str(col_data.mean()))
        print('Median: ' + str(col_data.median()))
        print('Standard Deviation: ' + str(col_data.std()))
        print('Min Value: ' + str(col_data.min()))
        print('Max Value: ' + str(col_data.max()))
