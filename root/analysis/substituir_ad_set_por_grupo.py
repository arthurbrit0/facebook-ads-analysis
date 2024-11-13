def substituir_ad_set_por_grupo(df, coluna_ad_set='Ad Set Name'):
    def classificar_grupo(ad_set_name):
        if isinstance(ad_set_name, str):
            ad_set_name = ad_set_name.upper()
            if ad_set_name.startswith('LC'):
                return 'Lookalike Conversion'
            elif ad_set_name.startswith('ADD TO CART'):
                return 'Add to cart'
            elif ad_set_name.startswith('VIEWED'):
                return 'Viewed'
            elif any(termo in ad_set_name for termo in ['EUROPE', 'W -', 'WW']):
                return 'Segmentação Demográfica e Geográfica'
            elif ad_set_name.startswith('INSTAGRAM POST'):
                return 'Instagram Campanha'
            elif ad_set_name.startswith('RL Cart-Conversion'):
                return 'Remarketing to add cart'
            else:
                return 'Outros'
        else:
            return 'Desconhecido'
    df['group_campanha'] = df[coluna_ad_set].apply(classificar_grupo)

    return df