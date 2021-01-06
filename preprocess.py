'''Importing required libraries'''

import pandas as pd
import numpy as np
np.random.seed(1)

'''Pre-processing data
Age            - replace nan with mode
Weight         - group by age and replace nan with median weight of corresponding age
Delivery phase - replace nan with mode
HB             - group by age and replace nan with median hb of corresponding age
BP             - replace nan with median
Education      - replace nan with mode
Residence      - group by community and replace nan with mode residence of corresponding community
'''

def preprocess(df):
    '''shuffling the rows'''
    df = df.sample(frac=1).reset_index(drop=True)
    '''removing the tuples that have more than 3 nan values'''
    df = df[df.isnull().sum(axis=1) < 4]
    df['Age'].fillna(df['Age'].mode()[0], inplace=True)
    df['Weight'] = df.groupby('Age')['Weight'].transform(lambda x: x.fillna(x.median()))
    df['Delivery phase'].fillna(df['Delivery phase'].mode()[0], inplace=True)
    df['HB'] = df['HB'].fillna(df.groupby('Age')['HB'].transform('median'))
    df['BP'] = df['BP'].fillna(df['BP'].median())
    df['Education'].fillna(df['Education'].mode()[0], inplace=True)
    df['Residence'] = df.groupby(['Community'], sort=False)['Residence'].apply(lambda x: x.fillna(x.mode().iloc[0]))
    normalize(df)
    return df

'''Min-Max Normalization for numerical columns - Age,Weight,HB,BP'''
def normalize(df):
    columns=['Age','Weight','HB','BP']
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

'''Load csv file into a dataframe'''
df = pd.read_csv (r'../data/LBW_Dataset.csv')
'''Preprocess'''
df = preprocess(df)
'''Write back the cleaned data to csv file'''
df.to_csv(r'../data/preprocessed.csv', index = False)
