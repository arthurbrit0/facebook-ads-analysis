import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from PIL import Image
from wordcloud import WordCloud
from nltk.corpus import stopwords

def excel_to_csv(excel_file):
    sheets = pd.read_excel(excel_file, sheet_name=None)
    path_output = '/home/arthurbrito/Downloads'

    for sheet_name, data in sheets.items():
        csv_file = os.path.join(path_output, f'{sheet_name}.csv')
        data.to_csv(csv_file, index=False)
        print(f'Sheet "{sheet_name}" salva como {csv_file}')

path_excel = '/home/arthurbrito/Downloads/Growth-Internship-Test.xlsx'
excel_to_csv(path_excel)

def csv_to_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    return df

PATH_BY_AGE = '/home/arthurbrito/Downloads/BY AGE.csv'
PATH_BY_COUNTRY = '/home/arthurbrito/Downloads/BY COUNTRY.csv'
PATH_BY_PLATFORM = '/home/arthurbrito/Downloads/BY PLATFORM.csv'

by_age_df = csv_to_dataframe(PATH_BY_AGE)
by_country_df = csv_to_dataframe(PATH_BY_COUNTRY)
by_platform_df = csv_to_dataframe(PATH_BY_PLATFORM)

by_age_df.head()

by_age_df.info()

by_age_unique = len(by_age_df['Ad Set Name'].unique())
print(by_age_unique)

by_country_df.head()

by_country_df.info()

by_county_unique = len(by_country_df['Ad Set Name'].unique())
print(by_county_unique)

by_platform_df.head()

by_platform_df.info()

by_platform_unique = len(by_platform_df['Ad Set Name'].unique())
print(by_platform_unique)

data_dictionary = {
    "Ad Set Name": {
        "dtype": "object",
        "description": "Nome do conjunto de anúncios."
    },
    "Country": {
        "dtype": "object",
        "description": "País onde os anúncios foram exibidos."
    },
    "Result Rate": {
        "dtype": "float64",
        "description": "Taxa de resultados gerados pelos anúncios."
    },
    "Result Indicator": {
        "dtype": "object",
        "description": "Indicador que representa o tipo de resultado obtido (ex: cliques, conversões)."
    },
    "Results": {
        "dtype": "int64",
        "description": "Número total de resultados alcançados com os anúncios."
    },
    "Reach": {
        "dtype": "int64",
        "description": "Número de pessoas únicas que visualizaram os anúncios."
    },
    "Frequency": {
        "dtype": "float64",
        "description": "Número médio de vezes que cada pessoa viu os anúncios."
    },
    "Link Clicks": {
        "dtype": "int64",
        "description": "Número total de cliques no link dos anúncios."
    },
    "CPC (Link) (USD)": {
        "dtype": "float64",
        "description": "Custo por clique em links, em dólares americanos."
    },
    "CPC (All) (USD)": {
        "dtype": "float64",
        "description": "Custo por clique em geral, em dólares americanos."
    },
    "Cost per 1,000 People Reached (USD)": {
        "dtype": "float64",
        "description": "Custo para alcançar 1.000 pessoas, em dólares americanos."
    },
    "CTR (All)": {
        "dtype": "float64",
        "description": "Taxa de cliques em relação ao número total de impressões."
    },
    "Add to Cart (Facebook Pixel)": {
        "dtype": "int64",
        "description": "Número de adições ao carrinho rastreadas pelo Facebook Pixel."
    },
    "Cost per Add To Cart (Facebook Pixel) (USD)": {
        "dtype": "float64",
        "description": "Custo por cada adição ao carrinho, em dólares americanos."
    },
    "Initiate Checkout (Facebook Pixel)": {
        "dtype": "int64",
        "description": "Número de inícios de checkout rastreados pelo Facebook Pixel."
    },
    "Cost per Initiate Checkout (Facebook Pixel) (USD)": {
        "dtype": "float64",
        "description": "Custo por cada início de checkout, em dólares americanos."
    },
    "Purchase (Facebook Pixel)": {
        "dtype": "int64",
        "description": "Número de compras rastreadas pelo Facebook Pixel."
    },
    "Cost per Purchase (Facebook Pixel) (USD)": {
        "dtype": "float64",
        "description": "Custo por cada compra, em dólares americanos."
    },
    "Amount Spent (USD)": {
        "dtype": "float64",
        "description": "Total gasto em anúncios, em dólares americanos."
    },
    "Purchase Conversion Value (Facebook Pixel)": {
        "dtype": "float64",
        "description": "Valor total das conversões de compras rastreadas pelo Facebook Pixel."
    }
}

def clean_dataframe(df):

    df = df.dropna()
    print("Linhas com valores NaN foram removidas.")

    duplicate_counts = df.duplicated().sum()
    if duplicate_counts > 0:
        print(f"Linhas duplicadas encontradas: {duplicate_counts}. Removendo linhas duplicadas.")
        df = df.drop_duplicates()
    else:
        print(f"Não há linhas duplicadas.")

    return df

by_age_df_cleaned = clean_dataframe(by_age_df)

by_country_df_cleaned = clean_dataframe(by_country_df)

by_platform_df_cleaned = clean_dataframe(by_platform_df)

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

by_age_df_refatorado = substituir_ad_set_por_grupo(by_age_df_cleaned)

by_age_df_refatorado = by_age_df_refatorado.rename(columns={
    "Platform": "platform",
    "Age": "age",
    "Result Rate": "result_rate",
    "Result Indicator": "result_indicator",
    "Results": "results",
    "Reach": "reach",
    "Frequency": "frequency",
    "Link Clicks": "link_clicks",
    "CPC (Link) (USD)": "cpc_link_usd",
    "CPC (All) (USD)": "cpc_all_usd",
    "Cost per 1,000 People Reached (USD)": "cpm_usd",
    "CTR (All)": "ctr_all",
    "Add to Cart (Facebook Pixel)": "add_to_cart",
    "Cost per Add To Cart (Facebook Pixel) (USD)": "cost_per_add_to_cart_usd",
    "Initiate Checkout (Facebook Pixel)": "initiate_checkout",
    "Cost per Initiate Checkout (Facebook Pixel) (USD)": "cost_per_initiate_checkout_usd",
    "Purchase (Facebook Pixel)": "purchase",
    "Cost per Purchase (Facebook Pixel) (USD)": "cost_per_purchase_usd",
    "Amount Spent (USD)": "amount_spent_usd",
    "Purchase Conversion Value (Facebook Pixel)": "purchase_conversion_value"
})[["group_campanha", "age", "result_rate", "result_indicator", "results", "reach", "frequency",
    "link_clicks", "cpc_link_usd", "cpc_all_usd", "cpm_usd", "ctr_all", "add_to_cart",
    "cost_per_add_to_cart_usd", "initiate_checkout", "cost_per_initiate_checkout_usd",
    "purchase", "cost_per_purchase_usd", "amount_spent_usd", "purchase_conversion_value"]]

by_age_df_refatorado.head()

by_age_df_refatorado.info()

by_country_df_refatorado = substituir_ad_set_por_grupo(by_country_df_cleaned)

by_country_df_refatorado = by_country_df_refatorado.rename(columns={
    "Country": "country",
    "Result Rate": "result_rate",
    "Result Indicator": "result_indicator",
    "Results": "results",
    "Reach": "reach",
    "Frequency": "frequency",
    "Link Clicks": "link_clicks",
    "CPC (Link) (USD)": "cpc_link_usd",
    "CPC (All) (USD)": "cpc_all_usd",
    "Cost per 1,000 People Reached (USD)": "cpm_usd",
    "CTR (All)": "ctr_all",
    "Add to Cart (Facebook Pixel)": "add_to_cart",
    "Cost per Add To Cart (Facebook Pixel) (USD)": "cost_per_add_to_cart_usd",
    "Initiate Checkout (Facebook Pixel)": "initiate_checkout",
    "Cost per Initiate Checkout (Facebook Pixel) (USD)": "cost_per_initiate_checkout_usd",
    "Purchase (Facebook Pixel)": "purchase",
    "Cost per Purchase (Facebook Pixel) (USD)": "cost_per_purchase_usd",
    "Amount Spent (USD)": "amount_spent_usd",
    "Purchase Conversion Value (Facebook Pixel)": "purchase_conversion_value"
})[["group_campanha", "country", "result_rate", "result_indicator", "results", "reach", "frequency",
    "link_clicks", "cpc_link_usd", "cpc_all_usd", "cpm_usd", "ctr_all", "add_to_cart",
    "cost_per_add_to_cart_usd", "initiate_checkout", "cost_per_initiate_checkout_usd",
    "purchase", "cost_per_purchase_usd", "amount_spent_usd", "purchase_conversion_value"]]

by_country_df_refatorado.head()

by_country_df_refatorado.info()

by_platform_df_refatorado = substituir_ad_set_por_grupo(by_platform_df_cleaned)

by_platform_df_refatorado = by_platform_df_refatorado.rename(columns={
    "Platform": "platform",
    "Result Rate": "result_rate",
    "Result Indicator": "result_indicator",
    "Results": "results",
    "Reach": "reach",
    "Frequency": "frequency",
    "Link Clicks": "link_clicks",
    "CPC (Link) (USD)": "cpc_link_usd",
    "CPC (All) (USD)": "cpc_all_usd",
    "Cost per 1,000 People Reached (USD)": "cpm_usd",
    "CTR (All)": "ctr_all",
    "Add to Cart (Facebook Pixel)": "add_to_cart",
    "Cost per Add To Cart (Facebook Pixel) (USD)": "cost_per_add_to_cart_usd",
    "Initiate Checkout (Facebook Pixel)": "initiate_checkout",
    "Cost per Initiate Checkout (Facebook Pixel) (USD)": "cost_per_initiate_checkout_usd",
    "Purchase (Facebook Pixel)": "purchase",
    "Cost per Purchase (Facebook Pixel) (USD)": "cost_per_purchase_usd",
    "Amount Spent (USD)": "amount_spent_usd",
    "Purchase Conversion Value (Facebook Pixel)": "purchase_conversion_value"
})[["group_campanha", "platform", "result_rate", "result_indicator", "results", "reach", "frequency",
    "link_clicks", "cpc_link_usd", "cpc_all_usd", "cpm_usd", "ctr_all", "add_to_cart",
    "cost_per_add_to_cart_usd", "initiate_checkout", "cost_per_initiate_checkout_usd",
    "purchase", "cost_per_purchase_usd", "amount_spent_usd", "purchase_conversion_value"]]

by_platform_df_refatorado.head()

by_platform_df_refatorado.info()

url_abv = '/home/arthurbrito/Downloads/country-by-abbreviation.json'
url_pop = '/home/arthurbrito/Downloads/country-by-population.json'

with open(url_abv, 'r') as file:
    siglas_json = json.load(file)

with open(url_pop, 'r') as file:
    populacao_json = json.load(file)

siglas_df = pd.DataFrame(siglas_json)
populacao_df  = pd.DataFrame(populacao_json)
if isinstance(populacao_json, list):
    populacao_json = {item['country']: item['population'] for item in populacao_json}

by_country_df_refatorado['country_name'] = by_country_df_refatorado['country'].map(siglas_df.set_index('abbreviation')['country'])

by_country_df_refatorado['population'] = by_country_df_refatorado['country_name'].map(populacao_json)

by_country_df_refatorado = by_country_df_refatorado[['group_campanha', 'country_name', 'population', 'result_rate', 'result_indicator', 'results', 'reach', 'frequency',
                                                     'link_clicks', 'cpc_link_usd', 'cpc_all_usd', 'cpm_usd', 'ctr_all', 'add_to_cart', 'cost_per_add_to_cart_usd',
                                                     'initiate_checkout', 'cost_per_initiate_checkout_usd', 'purchase', 'cost_per_purchase_usd', 'amount_spent_usd',
                                                     'purchase_conversion_value']]
by_country_df_refatorado.head(10)

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

by_age_df_std = std_dados(by_age_df_refatorado)
by_country_df_std = std_dados(by_country_df_refatorado)
by_platform_df_std = std_dados(by_platform_df_refatorado)

plataform_km = by_platform_df_std.drop(['group_campanha'], axis=1)

wcss = []

for n_clusters in range(1, 11):
    model = KMeans(n_clusters=n_clusters)

    model.fit(plataform_km)

    wcss.append(model.inertia_)

print(wcss)

model.__dict__

with sns.axes_style('whitegrid'):

  grafico = sns.lineplot(x=range(1, 11), y=wcss, marker="8", palette="pastel")
  grafico.set(title='Elbow Method', ylabel='WCSS', xlabel='Qtd. clusters')

def save_df_to_csv(df, file_name):
    folder_path = '/home/arthurbrito/Downloads'

    if not os.path.exists(folder_path):
        raise Exception(f"A pasta '{folder_path}' não existe.")

    file_path = os.path.join(folder_path, file_name)

    df.to_csv(file_path, index=False)

save_df_to_csv(by_age_df_refatorado, 'by_age_refatorado.csv')
save_df_to_csv(by_country_df_refatorado, 'by_country_refatorado.csv')
save_df_to_csv(by_platform_df_refatorado, 'by_platform_refatorado.csv')

save_df_to_csv(by_age_df_std, 'by_age_std.csv')
save_df_to_csv(by_country_df_std, 'by_country_std.csv')
save_df_to_csv(by_platform_df_std, 'by_platform_std.csv')

def plot_correlation_matrix(df, selected_columns, title='Matriz de Correlação'):
    correlation_matrix = df[selected_columns].corr()

    print(correlation_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

selected_columns = [
    'result_rate_std',
    'results_std',
    'reach_std',
    'frequency_std',
    'link_clicks_std',
    'ctr_all_std',
    'add_to_cart_std',
    'initiate_checkout_std',
    'purchase_std',
    'amount_spent_usd_std',
    'purchase_conversion_value_std'
]

plot_correlation_matrix(by_platform_df_std, selected_columns, title='Matriz de Correlação - BY_PLATFORM')

group_gereal = by_platform_df_refatorado.groupby('platform').agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

group_gereal['ROI (%)'] = ((group_gereal['purchase_conversion_value'] - group_gereal['amount_spent_usd']) / group_gereal['amount_spent_usd']) * 100

total_amount_spent = group_gereal['amount_spent_usd'].sum()
total_purchase_value = group_gereal['purchase_conversion_value'].sum()

group_gereal['percentage_amount_spent_usd'] = (group_gereal['amount_spent_usd'] / total_amount_spent) * 100
group_gereal['percentage_purchase_conversion_value'] = (group_gereal['purchase_conversion_value'] / total_purchase_value) * 100

group_gereal = group_gereal[['platform', 'amount_spent_usd', 'percentage_amount_spent_usd',
                                  'purchase_conversion_value', 'percentage_purchase_conversion_value',
                                  'ROI (%)']]
group_gereal.head()

group_gereal_platform = by_platform_df_refatorado.groupby(['platform', 'group_campanha']).agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

group_gereal_platform['ROI (%)'] = ((group_gereal_platform['purchase_conversion_value'] - group_gereal_platform['amount_spent_usd']) / group_gereal_platform['amount_spent_usd']) * 100

total_amount_spent = group_gereal_platform['amount_spent_usd'].sum()
total_purchase_value = group_gereal_platform['purchase_conversion_value'].sum()

group_gereal_platform['percentage_amount_spent_usd'] = (group_gereal_platform['amount_spent_usd'] / total_amount_spent) * 100
group_gereal_platform['percentage_purchase_conversion_value'] = (group_gereal_platform['purchase_conversion_value'] / total_purchase_value) * 100

group_gereal_platform['ROI (%)'] = group_gereal_platform['ROI (%)'].round(2)
group_gereal_platform['percentage_amount_spent_usd'] = group_gereal_platform['percentage_amount_spent_usd'].round(2)
group_gereal_platform['percentage_purchase_conversion_value'] = group_gereal_platform['percentage_purchase_conversion_value'].round(2)

group_gereal_platform = group_gereal_platform[['group_campanha', 'platform', 'amount_spent_usd',
                   'percentage_amount_spent_usd','purchase_conversion_value',
                   'percentage_purchase_conversion_value','ROI (%)']]
group_gereal_platform.head(15)

lookalike_conversion_df = by_platform_df_refatorado[by_platform_df_refatorado['group_campanha'] == 'Lookalike Conversion']

grouped_lookalike = lookalike_conversion_df.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grouped_lookalike['conversion_rate'] = (grouped_lookalike['purchase'] / grouped_lookalike['link_clicks']) * 100

grouped_lookalike['cpc_link_usd'] = grouped_lookalike['amount_spent_usd'] / grouped_lookalike['link_clicks']

grouped_lookalike['cpa_usd'] = grouped_lookalike['amount_spent_usd'] / grouped_lookalike['purchase']

grouped_lookalike['roi'] = ((grouped_lookalike['purchase_conversion_value'] - grouped_lookalike['amount_spent_usd']) / grouped_lookalike['amount_spent_usd']) * 100

total_amount_spent = grouped_lookalike['amount_spent_usd'].sum()
total_purchase_value = grouped_lookalike['purchase_conversion_value'].sum()

grouped_lookalike['percentage_amount_spent_usd'] = (grouped_lookalike['amount_spent_usd'] / total_amount_spent) * 100
grouped_lookalike['percentage_purchase_conversion_value'] = (grouped_lookalike['purchase_conversion_value'] / total_purchase_value) * 100

grouped_lookalike = grouped_lookalike.round(2)
grouped_lookalike.head()

import plotly.graph_objects as go

data = grouped_lookalike

colors = {
    'Facebook': 'blue',
    'Instagram': 'orange',
}

fig_reach = go.Figure()

for platform in data['platform'].unique():
    fig_reach.add_trace(go.Bar(
        x=[platform],
        y=data[data['platform'] == platform]['conversion_rate'],
        name='Taxa de conversão',
        marker=dict(color=colors.get(platform, 'grey')),  
        showlegend=False  
    ))

fig_reach.update_layout(
    title='Taxa de conversão por Plataforma',
    xaxis_title='Plataformas',
    yaxis_title='Taxa de conversão',
    width=600, 
    height=600, 
)

fig_reach.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike, x='platform', y='reach', palette='viridis')
plt.title('Total de alcance por Plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de alcance')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike, x='platform', y='amount_spent_usd', palette='viridis')
plt.title('Total de Compras por Plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike, x='platform', y='purchase', palette='viridis')
plt.title('Total de Compras por Plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike, x='platform', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por Plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('CPC (USD)')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=grouped_lookalike, x='reach', y='purchase', hue='platform', palette='viridis', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid()
plt.show()

lookalike_conversion_df = by_platform_df_refatorado[by_platform_df_refatorado['group_campanha'] == 'Lookalike Conversion']

grouped_lookalike_simu = lookalike_conversion_df.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grouped_lookalike_simu = grouped_lookalike_simu[grouped_lookalike_simu['platform'] != 'Audience Network']

grouped_lookalike_simu['cpm_usd'] = (grouped_lookalike_simu['amount_spent_usd'] / grouped_lookalike_simu['reach']) * 1000
investment = 10000

grouped_lookalike_simu['simulated_reach'] = (investment / grouped_lookalike_simu['cpm_usd']) * 1000

grouped_lookalike_simu['total_reach_with_investment'] = grouped_lookalike_simu['reach'] + grouped_lookalike_simu['simulated_reach']

grouped_lookalike_simu = grouped_lookalike_simu[['platform', 'group_campanha', 'simulated_reach', 'total_reach_with_investment']].round(2)

grouped_lookalike_simu.head()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike_simu, x='platform', y='simulated_reach', palette='viridis')
plt.title('Total de alcance por Plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de alcance')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

print(by_platform_df_refatorado.columns)

add_to_cart_df = by_platform_df_refatorado[by_platform_df_refatorado['group_campanha'] == 'Add to cart']

grouped_add_to_cart = add_to_cart_df.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grouped_add_to_cart['conversion_rate'] = (grouped_add_to_cart['purchase'] / grouped_add_to_cart['link_clicks']) * 100
grouped_add_to_cart['cpc_link_usd'] = grouped_add_to_cart['amount_spent_usd'] / grouped_add_to_cart['link_clicks']

grouped_add_to_cart['roi'] = ((grouped_add_to_cart['purchase_conversion_value'] - grouped_add_to_cart['amount_spent_usd']) / grouped_add_to_cart['amount_spent_usd']) * 100

total_amount_spent = grouped_add_to_cart['amount_spent_usd'].sum()
total_purchase_value = grouped_add_to_cart['purchase_conversion_value'].sum()

grouped_add_to_cart['percentage_amount_spent_usd'] = (grouped_add_to_cart['amount_spent_usd'] / total_amount_spent) * 100
grouped_add_to_cart['percentage_purchase_conversion_value'] = (grouped_add_to_cart['purchase_conversion_value'] / total_purchase_value) * 100

grouped_add_to_cart = grouped_add_to_cart.round(2)
grouped_add_to_cart.head()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart, x='platform', y='cpc_link_usd', palette='viridis')
plt.title('Total de Compras por Plataforma - Campanha "Add to Cart"')
plt.ylabel('Total de Compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart, x='platform', y='link_clicks', palette='viridis')
plt.title('Total de clicks por Plataforma - Campanha "Add to Cart"')
plt.ylabel('Total de clicks')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart, x='platform', y='cpc_link_usd', palette='viridis')
plt.title('Valor do click por Plataforma - Campanha "Add to Cart"')
plt.ylabel('Valor click')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart, x='platform', y='roi', palette='viridis')
plt.title('Retorno sobre Investimento (ROI) por Plataforma - Campanha "Add to Cart"')
plt.ylabel('ROI (%)')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

viewed_df = by_platform_df_refatorado[by_platform_df_refatorado['group_campanha'] == 'Viewed']

grouped_viewed = viewed_df.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'  
}).reset_index()

grouped_viewed['ctr_all'] = (grouped_viewed['link_clicks'] / grouped_viewed['reach']) * 100

grouped_viewed['cpm_usd'] = (grouped_viewed['amount_spent_usd'] / grouped_viewed['reach']) * 1000

grouped_viewed['cpc_link_usd'] = grouped_viewed['amount_spent_usd'] / grouped_viewed['link_clicks']

grouped_viewed['roi'] = ((grouped_viewed['purchase_conversion_value'] - grouped_viewed['amount_spent_usd']) / grouped_viewed['amount_spent_usd']) * 100

total_amount_spent = grouped_viewed['amount_spent_usd'].sum()
total_purchase_value = grouped_viewed['purchase_conversion_value'].sum()

grouped_viewed['percentage_amount_spent_usd'] = (grouped_viewed['amount_spent_usd'] / total_amount_spent) * 100
grouped_viewed['percentage_purchase_conversion_value'] = (grouped_viewed['purchase_conversion_value'] / total_purchase_value) * 100

grouped_viewed = grouped_viewed.round(2)
grouped_viewed.head()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed, x='platform', y='reach', palette='viridis')
plt.title('Alcance por Plataforma - Campanha Viewed')
plt.xlabel('Plataforma')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed, x='platform', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por Plataforma - Campanha Viewed')
plt.xlabel('Plataforma')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed, x='platform', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por Plataforma - Campanha Viewed')
plt.xlabel('Plataforma')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed, x='platform', y='roi', palette='viridis')
plt.title('ROI por Plataforma - Campanha Viewed')
plt.xlabel('Plataforma')
plt.ylabel('Retorno sobre Investimento (%)')
plt.grid(axis='y')
plt.show()

segmentation_df = by_platform_df_refatorado[by_platform_df_refatorado['group_campanha'] == 'Segmentação Demográfica e Geográfica']

grouped_segmentation = segmentation_df.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'mean',  
    'ctr_all': 'mean',    
    'cpm_usd': 'mean',    
    'cpc_link_usd': 'mean' 
}).reset_index()

grouped_segmentation['roi'] = ((grouped_segmentation['purchase_conversion_value'] - grouped_segmentation['amount_spent_usd']) / grouped_segmentation['amount_spent_usd']) * 100

total_amount_spent_segmentation = grouped_segmentation['amount_spent_usd'].sum()
total_purchase_value_segmentation = grouped_segmentation['purchase_conversion_value'].sum()

grouped_segmentation['percentage_amount_spent_usd'] = (grouped_segmentation['amount_spent_usd'] / total_amount_spent_segmentation) * 100
grouped_segmentation['percentage_purchase_conversion_value'] = (grouped_segmentation['purchase_conversion_value'] / total_purchase_value_segmentation) * 100

grouped_segmentation = grouped_segmentation.round(2)
grouped_segmentation.head()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation, x='platform', y='amount_spent_usd', palette='viridis')
plt.title('Amount spent - Segmentação Demográfica e Geográfica')
plt.xlabel('Plataforma')
plt.ylabel('Amount spent')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation, x='platform', y='frequency', palette='viridis')
plt.title('Frequência Média por Plataforma - Segmentação Demográfica e Geográfica')
plt.xlabel('Plataforma')
plt.ylabel('Frequência Média')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation, x='platform', y='ctr_all', palette='Greens_d')
plt.title('Taxa de Cliques (CTR) por Plataforma - Segmentação Demográfica e Geográfica')
plt.xlabel('Plataforma')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation, x='platform', y='reach', palette='viridis')
plt.title('Alcance por Plataforma - Segmentação Demográfica e Geográfica')
plt.xlabel('Plataforma')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation, x='platform', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por Plataforma - Segmentação Demográfica e Geográfica')
plt.xlabel('Plataforma')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

instagram_campaign_df = by_platform_df_refatorado[by_platform_df_refatorado['group_campanha'] == 'Instagram Campanha']

grouped_instagram = instagram_campaign_df.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'sum',  
    'ctr_all': 'mean',   
    'cpm_usd': 'mean',  
    'cpc_link_usd': 'mean'  
}).reset_index()

grouped_instagram['roi'] = ((grouped_instagram['purchase_conversion_value'] - grouped_instagram['amount_spent_usd']) / grouped_instagram['amount_spent_usd']) * 100

total_amount_spent_instagram = grouped_instagram['amount_spent_usd'].sum()
total_purchase_value_instagram = grouped_instagram['purchase_conversion_value'].sum()

grouped_instagram['percentage_amount_spent_usd'] = (grouped_instagram['amount_spent_usd'] / total_amount_spent_instagram) * 100
grouped_instagram['percentage_purchase_conversion_value'] = (grouped_instagram['purchase_conversion_value'] / total_purchase_value_instagram) * 100

grouped_instagram = grouped_instagram.round(2)

grouped_instagram.head()

plot_correlation_matrix(by_age_df_std, selected_columns, title='Matriz de Correlação - BY_AGE')

group_gereal_age = by_age_df_refatorado.groupby('age').agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

group_gereal_age['ROI (%)'] = ((group_gereal_age['purchase_conversion_value'] - group_gereal_age['amount_spent_usd']) / group_gereal_age['amount_spent_usd']) * 100

total_amount_spent = group_gereal_age['amount_spent_usd'].sum()
total_purchase_value = group_gereal_age['purchase_conversion_value'].sum()

group_gereal_age['percentage_amount_spent_usd'] = (group_gereal_age['amount_spent_usd'] / total_amount_spent) * 100
group_gereal_age['percentage_purchase_conversion_value'] = (group_gereal_age['purchase_conversion_value'] / total_purchase_value) * 100

group_gereal_age = group_gereal_age[['age', 'amount_spent_usd', 'percentage_amount_spent_usd',
                                  'purchase_conversion_value', 'percentage_purchase_conversion_value',
                                  'ROI (%)']]
group_gereal_age.head()

group_gereal_age = by_age_df_refatorado.groupby(['age', 'group_campanha']).agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

group_gereal_age['ROI (%)'] = ((group_gereal_age['purchase_conversion_value'] - group_gereal_age['amount_spent_usd']) / group_gereal_age['amount_spent_usd']) * 100

total_amount_spent = group_gereal_age['amount_spent_usd'].sum()
total_purchase_value = group_gereal_age['purchase_conversion_value'].sum()

group_gereal_age['percentage_amount_spent_usd'] = (group_gereal_age['amount_spent_usd'] / total_amount_spent) * 100
group_gereal_age['percentage_purchase_conversion_value'] = (group_gereal_age['purchase_conversion_value'] / total_purchase_value) * 100

group_gereal_age['ROI (%)'] = group_gereal_age['ROI (%)'].round(2)
group_gereal_age['percentage_amount_spent_usd'] = group_gereal_age['percentage_amount_spent_usd'].round(2)
group_gereal_age['percentage_purchase_conversion_value'] = group_gereal_age['percentage_purchase_conversion_value'].round(2)

group_gereal_age = group_gereal_age[['group_campanha', 'age', 'amount_spent_usd',
                   'percentage_amount_spent_usd','purchase_conversion_value',
                   'percentage_purchase_conversion_value','ROI (%)']]

group_gereal_age.head(37)

lookalike_conversion_age_df = by_age_df_refatorado[by_age_df_refatorado['group_campanha'] == 'Lookalike Conversion']

grouped_lookalike_age = lookalike_conversion_age_df.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grouped_lookalike_age['conversion_rate'] = (grouped_lookalike_age['purchase'] / grouped_lookalike_age['link_clicks']) * 100

grouped_lookalike_age['cpc_link_usd'] = grouped_lookalike_age['amount_spent_usd'] / grouped_lookalike_age['link_clicks']

grouped_lookalike_age['cpa_usd'] = grouped_lookalike_age['amount_spent_usd'] / grouped_lookalike_age['purchase']

grouped_lookalike_age['roi(%)'] = ((grouped_lookalike_age['purchase_conversion_value'] - grouped_lookalike_age['amount_spent_usd']) / grouped_lookalike_age['amount_spent_usd']) * 100

total_amount_spent = grouped_lookalike_age['amount_spent_usd'].sum()
total_purchase_value = grouped_lookalike_age['purchase_conversion_value'].sum()

grouped_lookalike_age['percentage_amount_spent_usd'] = (grouped_lookalike_age['amount_spent_usd'] / total_amount_spent) * 100
grouped_lookalike_age['percentage_purchase_conversion_value'] = (grouped_lookalike_age['purchase_conversion_value'] / total_purchase_value) * 100

grouped_lookalike_age = grouped_lookalike_age.round(2)
grouped_lookalike_age.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike_age, x='age', y='reach', palette='viridis')
plt.title('Total de alcance por idade - Campanha "Lookalike Conversion"')
plt.ylabel('Total de alcance')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike_age, x='age', y='amount_spent_usd', palette='viridis')
plt.title('Total de Compras por idade - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike_age, x='age', y='purchase', palette='viridis')
plt.title('Total de Compras por idade - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_lookalike_age, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por idade - Campanha "Lookalike Conversion"')
plt.ylabel('CPC (USD)')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=grouped_lookalike_age, x='reach', y='purchase', hue='age', palette='viridis', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid()
plt.show()

add_to_cart_age_df = by_age_df_refatorado[by_age_df_refatorado['group_campanha'] == 'Add to cart']

grouped_add_to_cart_age = add_to_cart_age_df.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grouped_add_to_cart_age['conversion_rate'] = (grouped_add_to_cart_age['purchase'] / grouped_add_to_cart_age['link_clicks']) * 100
grouped_add_to_cart_age['cpc_link_usd'] = grouped_add_to_cart_age['amount_spent_usd'] / grouped_add_to_cart_age['link_clicks']

grouped_add_to_cart_age['roi(%)'] = ((grouped_add_to_cart_age['purchase_conversion_value'] - grouped_add_to_cart_age['amount_spent_usd']) / grouped_add_to_cart_age['amount_spent_usd']) * 100

total_amount_spent = grouped_add_to_cart_age['amount_spent_usd'].sum()
total_purchase_value = grouped_add_to_cart_age['purchase_conversion_value'].sum()

grouped_add_to_cart_age['percentage_amount_spent_usd'] = (grouped_add_to_cart_age['amount_spent_usd'] / total_amount_spent) * 100
grouped_add_to_cart_age['percentage_purchase_conversion_value'] = (grouped_add_to_cart_age['purchase_conversion_value'] / total_purchase_value) * 100

grouped_add_to_cart_age = grouped_add_to_cart_age.round(2)

grouped_add_to_cart_age = grouped_add_to_cart_age[['age', 'group_campanha', 'reach', 'link_clicks', 'cpc_link_usd', 'purchase', 'conversion_rate','amount_spent_usd',
                                            'percentage_amount_spent_usd', 'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)' ]]
grouped_add_to_cart_age.head()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=grouped_lookalike_age, x='cpc_link_usd', y='link_clicks', hue='age', palette='viridis', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart_age, x='age', y='link_clicks', palette='viridis')
plt.title('Total de clicks por idade - Campanha "Add to Cart"')
plt.ylabel('Total de clicks')
plt.xlabel('idade')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart_age, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Valor do click por idade - Campanha "Add to Cart"')
plt.ylabel('Valor click')
plt.xlabel('idade')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_add_to_cart_age, x='age', y='roi(%)', palette='viridis')
plt.title('Retorno sobre Investimento (ROI) por idade - Campanha "Add to Cart"')
plt.ylabel('ROI (%)')
plt.xlabel('idade')
plt.grid(axis='y')
plt.show()

viewed_age_df = by_age_df_refatorado[by_age_df_refatorado['group_campanha'] == 'Viewed']

grouped_viewed_age = viewed_age_df.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'  
}).reset_index()

grouped_viewed_age['ctr_all'] = (grouped_viewed_age['link_clicks'] / grouped_viewed_age['reach']) * 100

grouped_viewed_age['cpm_usd'] = (grouped_viewed_age['amount_spent_usd'] / grouped_viewed_age['reach']) * 1000

grouped_viewed_age['cpc_link_usd'] = grouped_viewed_age['amount_spent_usd'] / grouped_viewed_age['link_clicks']

grouped_viewed_age['roi'] = ((grouped_viewed_age['purchase_conversion_value'] - grouped_viewed_age['amount_spent_usd']) / grouped_viewed_age['amount_spent_usd']) * 100

total_amount_spent = grouped_viewed_age['amount_spent_usd'].sum()
total_purchase_value = grouped_viewed_age['purchase_conversion_value'].sum()

grouped_viewed_age['percentage_amount_spent_usd'] = (grouped_viewed_age['amount_spent_usd'] / total_amount_spent) * 100
grouped_viewed_age['percentage_purchase_conversion_value'] = (grouped_viewed_age['purchase_conversion_value'] / total_purchase_value) * 100

grouped_viewed_age = grouped_viewed_age.round(2)
grouped_viewed_age.head()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed_age, x='age', y='reach', palette='viridis')
plt.title('Alcance por idade - Campanha Viewed')
plt.xlabel('Idade')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed_age, x='age', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por idade - Campanha Viewed')
plt.xlabel('idade')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed_age, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por idade - Campanha Viewed')
plt.xlabel('Idade')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_viewed_age, x='age', y='roi', palette='viridis')
plt.title('ROI por idade - Campanha Viewed')
plt.xlabel('Idade')
plt.ylabel('Retorno sobre Investimento (%)')
plt.grid(axis='y')
plt.show()

"""<h4>Segmentação Demográfica e Geográfica<h4>"""

segmentation_df_age = by_age_df_refatorado[by_age_df_refatorado['group_campanha'] == 'Segmentação Demográfica e Geográfica']

grouped_segmentation_age = segmentation_df_age.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'mean',  
    'ctr_all': 'mean',     
    'cpm_usd': 'mean',     
    'cpc_link_usd': 'mean' 
}).reset_index()

grouped_segmentation_age['roi'] = ((grouped_segmentation_age['purchase_conversion_value'] - grouped_segmentation_age['amount_spent_usd']) / grouped_segmentation_age['amount_spent_usd']) * 100

total_amount_spent_segmentation = grouped_segmentation_age['amount_spent_usd'].sum()
total_purchase_value_segmentation = grouped_segmentation_age['purchase_conversion_value'].sum()

grouped_segmentation_age['percentage_amount_spent_usd'] = (grouped_segmentation_age['amount_spent_usd'] / total_amount_spent_segmentation) * 100
grouped_segmentation_age['percentage_purchase_conversion_value'] = (grouped_segmentation_age['purchase_conversion_value'] / total_purchase_value_segmentation) * 100

grouped_segmentation_age = grouped_segmentation_age.round(2)
grouped_segmentation_age.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_age, x='age', y='amount_spent_usd', palette='viridis')
plt.title('Amount spent - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Amount spent')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_age, x='age', y='frequency', palette='viridis')
plt.title('Frequência Média por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Frequência Média')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_age, x='age', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_age, x='age', y='cpm_usd', palette='viridis')
plt.title('Custo médio por mil impressões - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Custo médio')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_age, x='age', y='reach', palette='viridis')
plt.title('Alcance por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_age, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

instagram_campaign_age_df = by_age_df_refatorado[by_age_df_refatorado['group_campanha'] == 'Instagram Campanha']

grouped_instagram_age = instagram_campaign_age_df.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'sum',  
    'ctr_all': 'mean',   
    'cpm_usd': 'mean',   
    'cpc_link_usd': 'mean'  
}).reset_index()

grouped_instagram_age['roi'] = ((grouped_instagram_age['purchase_conversion_value'] - grouped_instagram_age['amount_spent_usd']) / grouped_instagram_age['amount_spent_usd']) * 100

total_amount_spent_instagram = grouped_instagram_age['amount_spent_usd'].sum()
total_purchase_value_instagram = grouped_instagram_age['purchase_conversion_value'].sum()

grouped_instagram_age['percentage_amount_spent_usd'] = (grouped_instagram_age['amount_spent_usd'] / total_amount_spent_instagram) * 100
grouped_instagram_age['percentage_purchase_conversion_value'] = (grouped_instagram_age['purchase_conversion_value'] / total_purchase_value_instagram) * 100

grouped_instagram_age = grouped_instagram_age.round(2)

grouped_instagram_age.head()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_instagram_age, x='age', y='reach', palette='viridis')
plt.title('reach - Instagram')
plt.xlabel('Idade')
plt.ylabel('Custo médio')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_instagram_age, x='age', y='cpm_usd', palette='viridis')
plt.title('Alcance - Instagram')
plt.xlabel('Idade')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plot_correlation_matrix(by_country_df_std, selected_columns, title='Matriz de Correlação - BY_COUNTRY')

print(by_country_df_refatorado.columns)

by_country_df_refatorado.head(20)

lookalike_conversion_country_df = by_country_df_refatorado[by_country_df_refatorado['group_campanha'] == 'Lookalike Conversion']

grouped_lookalike_country = lookalike_conversion_country_df.groupby('country_name').agg({
    'result_rate': 'sum',
    'population': 'mean',
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grouped_lookalike_country['conversion_rate (%)'] = (grouped_lookalike_country['purchase'] / grouped_lookalike_country['link_clicks']) * 100

grouped_lookalike_country['cpc_link_usd'] = grouped_lookalike_country['amount_spent_usd'] / grouped_lookalike_country['link_clicks']

grouped_lookalike_country['cpa_usd'] = grouped_lookalike_country['amount_spent_usd'] / grouped_lookalike_country['purchase']

grouped_lookalike_country['reach_per_population (%)'] = (grouped_lookalike_country['reach'] / grouped_lookalike_country['population']) * 100

grouped_lookalike_country['ROI (%)'] = ((grouped_lookalike_country['purchase_conversion_value'] - grouped_lookalike_country['amount_spent_usd']) / grouped_lookalike_country['amount_spent_usd']) * 100

total_amount_spent = grouped_lookalike_country['amount_spent_usd'].sum()
total_purchase_value = grouped_lookalike_country['purchase_conversion_value'].sum()

grouped_lookalike_country['percentage_amount_spent_usd'] = (grouped_lookalike_country['amount_spent_usd'] / total_amount_spent) * 100
grouped_lookalike_country['percentage_purchase_conversion_value'] = (grouped_lookalike_country['purchase_conversion_value'] / total_purchase_value) * 100

grouped_lookalike_country = grouped_lookalike_country.loc[grouped_lookalike_country['ROI (%)'] > -100.00]

grouped_lookalike_country = grouped_lookalike_country.dropna()

final_columns = [
    'country_name', 'population', 'reach_per_population (%)', 'result_rate', 'conversion_rate (%)', 'link_clicks', 'cpc_link_usd',
    'cpa_usd', 'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'ROI (%)'
]
grouped_lookalike_country = grouped_lookalike_country[final_columns].round(2).sort_values(by='reach_per_population (%)', ascending=False).reset_index(drop=True)

grouped_lookalike_country.head(10)

top_10 = grouped_lookalike_country.head(10)

plt.figure(figsize=(15, 8))
sns.scatterplot(data=top_10, x='purchase_conversion_value', y='reach_per_population (%)', hue='country_name', palette='plasma', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10, x='country_name', y='reach_per_population (%)', palette='viridis')
plt.title('Alcance/População(%) - País')
plt.xlabel('País')
plt.ylabel('Alcance/População(%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10, x='country_name', y='result_rate', palette='viridis')
plt.title('result_rate - País')
plt.xlabel('País')
plt.ylabel('result_rate')
plt.grid(axis='y')
plt.show()


plt.figure(figsize=(15, 8))
sns.barplot(data=top_10, x='country_name', y='amount_spent_usd', palette='viridis')
plt.title('amount_spent_usd - País')
plt.xlabel('País')
plt.ylabel('amount_spent_usd')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10, x='country_name', y='conversion_rate (%)', palette='viridis')
plt.title('conversion_rate - País')
plt.xlabel('País')
plt.ylabel('conversion_rate')
plt.grid(axis='y')
plt.show()

add_to_cart_country_df = by_country_df_refatorado[by_country_df_refatorado['group_campanha'] == 'Add to cart']

grouped_add_to_cart_country = add_to_cart_country_df.groupby(['country_name', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'population': 'mean'
}).reset_index()

grouped_add_to_cart_country['conversion_rate'] = (grouped_add_to_cart_country['purchase'] / grouped_add_to_cart_country['link_clicks']) * 100

grouped_add_to_cart_country['cpc_link_usd'] = grouped_add_to_cart_country['amount_spent_usd'] / grouped_add_to_cart_country['link_clicks']

grouped_add_to_cart_country['reach_per_population (%)'] = (grouped_add_to_cart_country['reach'] / grouped_add_to_cart_country['population']) * 100

grouped_add_to_cart_country['roi(%)'] = ((grouped_add_to_cart_country['purchase_conversion_value'] - grouped_add_to_cart_country['amount_spent_usd']) / grouped_add_to_cart_country['amount_spent_usd']) * 100

total_amount_spent = grouped_add_to_cart_country['amount_spent_usd'].sum()
total_purchase_value = grouped_add_to_cart_country['purchase_conversion_value'].sum()

grouped_add_to_cart_country['percentage_amount_spent_usd'] = (grouped_add_to_cart_country['amount_spent_usd'] / total_amount_spent) * 100
grouped_add_to_cart_country['percentage_purchase_conversion_value'] = (grouped_add_to_cart_country['purchase_conversion_value'] / total_purchase_value) * 100

grouped_add_to_cart_country = grouped_add_to_cart_country.loc[grouped_add_to_cart_country['roi(%)'] > -100.00]

grouped_add_to_cart_country = grouped_add_to_cart_country.dropna()

final_columns = [
    'country_name', 'population', 'reach_per_population (%)', 'conversion_rate', 'link_clicks', 'cpc_link_usd',
    'purchase', 'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)'
]

grouped_add_to_cart_country = grouped_add_to_cart_country[final_columns].round(2).sort_values(
    by=['purchase_conversion_value'], ascending=False 
).reset_index(drop=True)

grouped_add_to_cart_country.head(10)

top_10_add_to_cart = grouped_add_to_cart_country.head(10)

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10_add_to_cart, x='country_name', y='conversion_rate', palette='viridis')
plt.title('Taxa de Conversão por País')
plt.ylabel('Taxa de Conversão (%)')
plt.xlabel('País')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10_add_to_cart, x='country_name', y='link_clicks', palette='viridis')
plt.title('Total de clicks por País - Campanha "Add to Cart"')
plt.ylabel('Total de clicks')
plt.xlabel('País')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.scatterplot(data=top_10_add_to_cart, x='cpc_link_usd', y='link_clicks', hue='country_name', palette='viridis', s=100)
plt.title('CPC vs Clicks por País')
plt.ylabel('Número de Cliques')
plt.xlabel('Custo por Clique (USD)')
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10_add_to_cart, x='country_name', y='purchase', palette='viridis')
plt.title('Purchase - Campanha "Add to Cart"')
plt.ylabel('Purchase')
plt.xlabel('País')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10_add_to_cart, x='country_name', y='cpc_link_usd', palette='viridis')
plt.title('Valor do click por País - Campanha "Add to Cart"')
plt.ylabel('Valor click')
plt.xlabel('País')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=top_10_add_to_cart, x='country_name', y='roi(%)', palette='viridis')
plt.title('Retorno sobre Investimento (ROI) por País - Campanha "Add to Cart"')
plt.ylabel('roi(%)')
plt.xlabel('País')
plt.grid(axis='y')
plt.show()

viewed_country_df = by_country_df_refatorado[by_country_df_refatorado['group_campanha'] == 'Viewed']

grouped_viewed_country = viewed_country_df.groupby(['country_name', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'population': 'mean',
    'purchase_conversion_value': 'sum',
    'purchase': 'sum'
}).reset_index()

grouped_viewed_country['ctr_all'] = (grouped_viewed_country['link_clicks'] / grouped_viewed_country['reach']) * 100

grouped_viewed_country['cpm_usd'] = (grouped_viewed_country['amount_spent_usd'] / grouped_viewed_country['reach']) * 1000

grouped_viewed_country['reach_per_population (%)'] = (grouped_viewed_country['reach'] / grouped_viewed_country['population']) * 100

grouped_viewed_country['cpc_link_usd'] = grouped_viewed_country['amount_spent_usd'] / grouped_viewed_country['link_clicks']

grouped_viewed_country['roi(%)'] = ((grouped_viewed_country['purchase_conversion_value'] - grouped_viewed_country['amount_spent_usd']) / grouped_viewed_country['amount_spent_usd']) * 100
grouped_viewed_country['conversion_rate'] = (grouped_viewed_country['purchase'] / grouped_viewed_country['link_clicks']) * 100

total_amount_spent = grouped_viewed_country['amount_spent_usd'].sum()
total_purchase_value = grouped_viewed_country['purchase_conversion_value'].sum()

grouped_viewed_country['percentage_amount_spent_usd'] = (grouped_viewed_country['amount_spent_usd'] / total_amount_spent) * 100
grouped_viewed_country['percentage_purchase_conversion_value'] = (grouped_viewed_country['purchase_conversion_value'] / total_purchase_value) * 100

grouped_viewed_country = grouped_viewed_country.loc[grouped_viewed_country['roi(%)'] > -100.00]

grouped_viewed_country = grouped_viewed_country.dropna()

final_columns = [
    'country_name', 'population', 'reach_per_population (%)', 'conversion_rate', 'ctr_all', 'link_clicks', 'cpc_link_usd', 'cpm_usd',
    'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)'
]

grouped_viewed_country = grouped_viewed_country[final_columns].round(2).sort_values(
    by=['reach_per_population (%)'], ascending=[False]  
).reset_index(drop=True)

grouped_viewed_country.head(10)

top_10 = grouped_viewed_country.head(10)

plt.figure(figsize=(15, 8))
sns.scatterplot(data=top_10, x='purchase_conversion_value', y='reach_per_population (%)', hue='country_name', palette='plasma', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid(axis='y')
plt.show()

top_10_viewed = grouped_viewed_country.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_viewed, x='country_name', y='reach_per_population (%)', palette='viridis')
plt.title('Alcance por País - Campanha Viewed')
plt.xlabel('País')
plt.ylabel('Alcance por população(%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_viewed, x='country_name', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por País - Campanha Viewed')
plt.xlabel('País')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_viewed, x='country_name', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por País - Campanha Viewed')
plt.xlabel('País')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_viewed, x='country_name', y='cpm_usd', palette='viridis')
plt.title('Custo por Mil Impressões - Campanha Viewed')
plt.xlabel('País')
plt.ylabel('Custo por Mil Impressões')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_viewed, x='country_name', y='roi(%)', palette='viridis')
plt.title('ROI por idade - Campanha Viewed')
plt.xlabel('País')
plt.ylabel('Retorno sobre Investimento (%)')
plt.grid(axis='y')
plt.show()

segmentation_df_country = by_country_df_refatorado[by_country_df_refatorado['group_campanha'] == 'Segmentação Demográfica e Geográfica']

grouped_segmentation_country = segmentation_df_country.groupby(['country_name', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'mean',
    'ctr_all': 'mean',
    'cpm_usd': 'mean',
    'cpc_link_usd': 'mean',
    'population': 'mean'
}).reset_index()


grouped_segmentation_country['roi(%)'] = ((grouped_segmentation_country['purchase_conversion_value'] - grouped_segmentation_country['amount_spent_usd']) / grouped_segmentation_country['amount_spent_usd']) * 100

grouped_segmentation_country['reach_per_population (%)'] = (grouped_segmentation_country['reach'] / grouped_segmentation_country['population']) * 100

total_amount_spent_segmentation = grouped_segmentation_country['amount_spent_usd'].sum()
total_purchase_value_segmentation = grouped_segmentation_country['purchase_conversion_value'].sum()

grouped_segmentation_country['percentage_amount_spent_usd'] = (grouped_segmentation_country['amount_spent_usd'] / total_amount_spent_segmentation) * 100
grouped_segmentation_country['percentage_purchase_conversion_value'] = (grouped_segmentation_country['purchase_conversion_value'] / total_purchase_value_segmentation) * 100

grouped_segmentation_country = grouped_segmentation_country.loc[grouped_segmentation_country['roi(%)'] > -100.00]
grouped_segmentation_country = grouped_segmentation_country.dropna()

final_columns = [
    'country_name', 'population', 'frequency', 'reach_per_population (%)', 'link_clicks', 'cpc_link_usd', 'cpm_usd',
    'ctr_all', 'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)'
]

grouped_segmentation_country = grouped_segmentation_country[final_columns].round(2).sort_values(
    by=['reach_per_population (%)'], ascending=[False]  
).reset_index(drop=True)

grouped_segmentation_country.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_country, x='country_name', y='amount_spent_usd', palette='viridis')
plt.title('Amount spent - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Amount spent')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_country, x='country_name', y='reach_per_population (%)', palette='viridis')
plt.title('Alcande por população(%) - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Alcande por população(%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_country, x='country_name', y='frequency', palette='viridis')
plt.title('Frequência Média por País - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Frequência Média')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_country, x='country_name', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por País - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_country, x='country_name', y='cpm_usd', palette='viridis')
plt.title('Custo médio por mil impressões - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Custo médio')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_segmentation_country, x='country_name', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por País - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

instagram_campaign_country_df = by_country_df_refatorado[by_country_df_refatorado['group_campanha'] == 'Instagram Campanha']

grouped_instagram_country = instagram_campaign_country_df.groupby(['country_name', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'sum',  
    'ctr_all': 'mean',   
    'cpm_usd': 'mean',   
    'cpc_link_usd': 'mean',  
    'population': 'mean'
}).reset_index()

grouped_instagram_country['roi(%)'] = ((grouped_instagram_country['purchase_conversion_value'] - grouped_instagram_country['amount_spent_usd']) / grouped_instagram_country['amount_spent_usd']) * 100

total_amount_spent_instagram = grouped_instagram_country['amount_spent_usd'].sum()
total_purchase_value_instagram = grouped_instagram_country['purchase_conversion_value'].sum()

grouped_instagram_country['percentage_amount_spent_usd'] = (grouped_instagram_country['amount_spent_usd'] / total_amount_spent_instagram) * 100
grouped_instagram_country['percentage_purchase_conversion_value'] = (grouped_instagram_country['purchase_conversion_value'] / total_purchase_value_instagram) * 100

grouped_instagram_country = grouped_instagram_country.round(2)

grouped_instagram_country.head()

text = " ".join(by_age_df['Ad Set Name'].dropna().astype(str))

wc = WordCloud().generate(text)

wc = WordCloud(background_color='white', colormap = 'binary',
     stopwords = ['meta'], width = 800, height = 500).generate(text)
plt.axis("off")
plt.imshow(wc)

incidencia_age = by_age_df_refatorado['group_campanha'].value_counts()
incidencia_country = by_country_df_refatorado['group_campanha'].value_counts()
incidencia_platform = by_platform_df_refatorado['group_campanha'].value_counts()

incidencia_total = pd.DataFrame({
    'group_campanha': incidencia_age.index,
    'age_count': incidencia_age.values,
    'country_count': incidencia_country.reindex(incidencia_age.index, fill_value=0).values,
    'platform_count': incidencia_platform.reindex(incidencia_age.index, fill_value=0).values
})

incidencia_total['mean_count'] = incidencia_total[['age_count', 'country_count', 'platform_count']].mean(axis=1).round(2)

grupo_labels = incidencia_total['group_campanha'].astype(str) 
grupo_valores = incidencia_total['mean_count'] 

fig = go.Figure(data=[
    go.Bar(
        x=grupo_labels,
        y=grupo_valores,
        marker_color='lightskyblue',
        text=grupo_valores,
        textposition='outside',
    )
])

fig.update_layout(
   title='Distribuição Média das Campanhas após Agrupamento',
    title_font=dict(size=20, color='black'),
    xaxis_title='Tipo de Campanha',
    yaxis_title='Número de Campanhas',
    xaxis_title_font=dict(size=14, color='black'),
    yaxis_title_font=dict(size=14, color='black'),
    plot_bgcolor='rgba(240, 240, 240, 0.8)',
    barmode='group',
    margin=dict(l=40, r=40, t=40, b=40)
)

fig.show()

data = {
    'country': ['Bélgica', 'Itália', 'Irlanda do Norte'],
    'investment': [4500, 3500, 2000],
}

df = pd.DataFrame(data)

df['continent'] = 'Europa'
df['iso_alpha'] = ['BEL', 'ITA', 'GBR'] 

fig = px.treemap(df, path=[px.Constant("Mundo"), 'continent', 'country'],
                   values='investment', color='investment',
                   hover_data=['iso_alpha'],
                   color_continuous_scale='RdBu',
                   color_continuous_midpoint=df['investment'].mean())

fig.update_traces(texttemplate="%{label}: %{value}", textfont_size=12)

fig.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),
    title='Investimentos por País',
)

fig.show()

data = {
    'plataforma': ['Instagram', 'Facebook'],
    'investment': [3000, 7000],
}

df = pd.DataFrame(data)

df['categoria'] = 'Plataforma'
df['iso_alpha'] = ['INS', 'FB'] 

fig = px.treemap(df, path=[px.Constant("Internet"), 'categoria', 'plataforma'],
                   values='investment', color='investment',
                   hover_data=['iso_alpha'],
                   color_continuous_scale='RdBu',
                   color_continuous_midpoint=df['investment'].mean())

fig.update_traces(texttemplate="%{label}: %{value}", textfont_size=12)

fig.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),
    title='Investimentos por Plataforma',
)

fig.show()

data = {
    'faixa_etaria': ['18-24', '25-34', '35-44'],
    'investment': [4000, 4000, 1000],
}

df = pd.DataFrame(data)

df['categoria'] = 'Idade'
df['iso_alpha'] = ['18-24', '25-34', '35-44']  

fig = px.treemap(df, path=[px.Constant("Mundo"), 'categoria', 'faixa_etaria'],
                   values='investment', color='investment',
                   hover_data=['iso_alpha'],
                   color_continuous_scale='RdBu',
                   color_continuous_midpoint=df['investment'].mean())

fig.update_traces(texttemplate="%{label}: %{value}", textfont_size=12)

fig.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),
    title='Investimentos por Faixa Etária',

)

fig.show()