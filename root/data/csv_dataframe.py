import pandas as pd

def csv_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    return df