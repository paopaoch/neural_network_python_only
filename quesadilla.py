import pandas as pd

def normalization(df, columns=list()):
    if len(columns) == 0:
        cols = list(df.columns)
    else:
        cols = columns
    for col in cols:
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    return df