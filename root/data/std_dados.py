def std_dados(df):
    df_normalizado = df.copy()

    selected_columns = [
    'result_rate',
    'results',
    'reach',
    'frequency',
    'link_clicks',
    'ctr_all',
    'add_to_cart',
    'initiate_checkout',
    'purchase',
    'amount_spent_usd',
    'purchase_conversion_value'
]

    for col in selected_columns:
        if col in df.columns:
            media = df[col].mean()
            desvio_padrao = df[col].std()
            df_normalizado[col + '_std'] = (df[col] - media) / desvio_padrao
    colunas_normalizadas = ['group_campanha'] + [col + '_std' for col in selected_columns if col in df.columns]

    df_normalizado = df_normalizado[colunas_normalizadas]

    return df_normalizado