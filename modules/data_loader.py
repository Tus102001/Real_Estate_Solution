import pandas as pd

def load_data(path='data/final.csv'):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise Exception("Data file not found. Please upload 'final.csv' in the data folder.")
