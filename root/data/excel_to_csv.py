import pandas as pd
import os

def excel_csv(excel_file):
    sheets = pd.read_excel(excel_file, sheet_name=None)
    path_output = '/home/arthurbrito/Downloads'

    for sheet_name, data in sheets.items():
        csv_file = os.path.join(path_output, f'{sheet_name}.csv')
        data.to_csv(csv_file, index=False)
        print(f'Sheet "{sheet_name}" salva como {csv_file}')