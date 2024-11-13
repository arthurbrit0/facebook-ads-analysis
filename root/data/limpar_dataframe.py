def limpar_dataframe(df):
    df = df.dropna()
    print("Linhas com valores inválidos foram removidas.")
    duplicate_counts = df.duplicated().sum()
    if duplicate_counts > 0:
        print(f"Linhas duplicadas encontradas: {duplicate_counts}. Removendo linhas duplicadas.")
        df = df.drop_duplicates()
    else:
        print(f"Não há linhas duplicadas.")
    return df