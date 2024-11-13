import os

def save_df_to_csv(df, file_name):
    folder_path = '/home/arthurbrito/Downloads'

    if not os.path.exists(folder_path):
        raise Exception(f"A pasta '{folder_path}' não existe.")

    file_path = os.path.join(folder_path, file_name)

    df.to_csv(file_path, index=False)