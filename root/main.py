import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from data.excel_to_csv import excel_csv
from data.csv_dataframe import csv_dataframe
from data.limpar_dataframe import limpar_dataframe
from analysis.substituir_ad_set_por_grupo import substituir_ad_set_por_grupo
from data.save_df_to_csv import save_df_to_csv
from data.std_dados import std_dados
from visualization.correlation_matrix import plot_correlation_matrix
import plotly.graph_objects as go

path_excel = '/home/arthurbrito/Downloads/Growth-Internship-Test.xlsx'
excel_csv(path_excel)

PATH_BY_AGE = '/home/arthurbrito/Downloads/BY AGE.csv'
PATH_BY_COUNTRY = '/home/arthurbrito/Downloads/BY COUNTRY.csv'
PATH_BY_PLATFORM = '/home/arthurbrito/Downloads/BY PLATFORM.csv'

idade_dataframe = csv_dataframe(PATH_BY_AGE)
pais_dataframe = csv_dataframe(PATH_BY_COUNTRY)
plataforma_dataframe = csv_dataframe(PATH_BY_PLATFORM)

idade_dataframe.head()
idade_dataframe.info()

idade_unica = len(idade_dataframe['Ad Set Name'].unique())
print(idade_unica)

pais_dataframe.head()
pais_dataframe.info()

regiao_unica = len(pais_dataframe['Ad Set Name'].unique())
print(regiao_unica)

plataforma_dataframe.head()
plataforma_dataframe.info()

plataforma_unica = len(plataforma_dataframe['Ad Set Name'].unique())
print(plataforma_unica)

idade_dataframe_limpo = limpar_dataframe(idade_dataframe)
pais_dataframe_limpo = limpar_dataframe(pais_dataframe)
plataforma_dataframe_limpo = limpar_dataframe(plataforma_dataframe)

idade_dataframe_ref = substituir_ad_set_por_grupo(idade_dataframe_limpo)

idade_dataframe_ref = idade_dataframe_ref.rename(columns={
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

idade_dataframe_ref.head()

idade_dataframe_ref.info()

pais_dataframe_ref = substituir_ad_set_por_grupo(pais_dataframe_limpo)

pais_dataframe_ref = pais_dataframe_ref.rename(columns={
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

pais_dataframe_ref.head()

pais_dataframe_ref.info()

plataforma_dataframe_ref = substituir_ad_set_por_grupo(plataforma_dataframe_limpo)

plataforma_dataframe_ref = plataforma_dataframe_ref.rename(columns={
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

plataforma_dataframe_ref.head()

plataforma_dataframe_ref.info()

url_abreviado = '/home/arthurbrito/Downloads/country-by-abbreviation.json'
url_populacao = '/home/arthurbrito/Downloads/country-by-population.json'

with open(url_abreviado, 'r') as file:
    siglas_json = json.load(file)

with open(url_populacao, 'r') as file:
    populacao_json = json.load(file)

dataframe_siglas = pd.DataFrame(siglas_json)
dataframe_populacao  = pd.DataFrame(populacao_json)
if isinstance(populacao_json, list):
    populacao_json = {item['country']: item['population'] for item in populacao_json}

pais_dataframe_ref['country_name'] = pais_dataframe_ref['country'].map(dataframe_siglas.set_index('abbreviation')['country'])
pais_dataframe_ref['population'] = pais_dataframe_ref['country_name'].map(populacao_json)
pais_dataframe_ref = pais_dataframe_ref[['group_campanha', 'country_name', 'population', 'result_rate', 'result_indicator', 'results', 'reach', 'frequency','link_clicks', 'cpc_link_usd', 'cpc_all_usd', 'cpm_usd', 'ctr_all', 'add_to_cart', 'cost_per_add_to_cart_usd','initiate_checkout', 'cost_per_initiate_checkout_usd', 'purchase', 'cost_per_purchase_usd', 'amount_spent_usd','purchase_conversion_value']]
pais_dataframe_ref.head(10)

idade_dataframe_normal = std_dados(idade_dataframe_ref)
pais_dataframe_normal = std_dados(pais_dataframe_ref)
plataforma_dataframe_normal = std_dados(plataforma_dataframe_ref)

plataform_km = plataforma_dataframe_normal.drop(['group_campanha'], axis=1)

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

save_df_to_csv(idade_dataframe_ref, 'by_age_refatorado.csv')
save_df_to_csv(pais_dataframe_ref, 'by_country_refatorado.csv')
save_df_to_csv(plataforma_dataframe_ref, 'by_platform_refatorado.csv')

save_df_to_csv(idade_dataframe_normal, 'by_age_std.csv')
save_df_to_csv(pais_dataframe_normal, 'by_country_std.csv')
save_df_to_csv(plataforma_dataframe_normal, 'by_platform_std.csv')

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

plot_correlation_matrix(plataforma_dataframe_normal, selected_columns, title='Matriz de correlação por plataforma')

grupo_geral = plataforma_dataframe_ref.groupby('platform').agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

grupo_geral['ROI (%)'] = ((grupo_geral['purchase_conversion_value'] - grupo_geral['amount_spent_usd']) / grupo_geral['amount_spent_usd']) * 100

qtd_total_gasto = grupo_geral['amount_spent_usd'].sum()
valor_total_compra = grupo_geral['purchase_conversion_value'].sum()

grupo_geral['percentage_amount_spent_usd'] = (grupo_geral['amount_spent_usd'] /qtd_total_gasto) * 100
grupo_geral['percentage_purchase_conversion_value'] = (grupo_geral['purchase_conversion_value']/valor_total_compra) *100

grupo_geral = grupo_geral[['platform', 'amount_spent_usd', 'percentage_amount_spent_usd','purchase_conversion_value', 'percentage_purchase_conversion_value','ROI (%)']]
grupo_geral.head()

plataforma_grupo_geral = plataforma_dataframe_ref.groupby(['platform', 'group_campanha']).agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

plataforma_grupo_geral['ROI (%)'] = ((plataforma_grupo_geral['purchase_conversion_value'] - plataforma_grupo_geral['amount_spent_usd']) / plataforma_grupo_geral['amount_spent_usd']) * 100

qtd_total_gasto = plataforma_grupo_geral['amount_spent_usd'].sum()
valor_total_compra = plataforma_grupo_geral['purchase_conversion_value'].sum()

plataforma_grupo_geral['percentage_amount_spent_usd'] = (plataforma_grupo_geral['amount_spent_usd'] / qtd_total_gasto) * 100
plataforma_grupo_geral['percentage_purchase_conversion_value'] = (plataforma_grupo_geral['purchase_conversion_value'] / valor_total_compra) * 100

plataforma_grupo_geral['ROI (%)'] = plataforma_grupo_geral['ROI (%)'].round(2)
plataforma_grupo_geral['percentage_amount_spent_usd'] = plataforma_grupo_geral['percentage_amount_spent_usd'].round(2)
plataforma_grupo_geral['percentage_purchase_conversion_value'] = plataforma_grupo_geral['percentage_purchase_conversion_value'].round(2)

plataforma_grupo_geral = plataforma_grupo_geral[['group_campanha', 'platform', 'amount_spent_usd',
                   'percentage_amount_spent_usd','purchase_conversion_value',
                   'percentage_purchase_conversion_value','ROI (%)']]
plataforma_grupo_geral.head(15)

dataframe_convertido_lookalike = plataforma_dataframe_ref[plataforma_dataframe_ref['group_campanha'] == 'Lookalike Conversion']

lookalike_agrupado = dataframe_convertido_lookalike.groupby(['platform', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

lookalike_agrupado['conversion_rate'] = (lookalike_agrupado['purchase'] / lookalike_agrupado['link_clicks']) * 100
lookalike_agrupado['cpc_link_usd'] = lookalike_agrupado['amount_spent_usd'] / lookalike_agrupado['link_clicks']
lookalike_agrupado['cpa_usd'] = lookalike_agrupado['amount_spent_usd'] / lookalike_agrupado['purchase']
lookalike_agrupado['roi'] = ((lookalike_agrupado['purchase_conversion_value'] - lookalike_agrupado['amount_spent_usd']) / lookalike_agrupado['amount_spent_usd']) * 100

qtd_total_gasto = lookalike_agrupado['amount_spent_usd'].sum()
valor_total_compra = lookalike_agrupado['purchase_conversion_value'].sum()

lookalike_agrupado['percentage_amount_spent_usd'] = (lookalike_agrupado['amount_spent_usd'] / qtd_total_gasto) * 100
lookalike_agrupado['percentage_purchase_conversion_value'] = (lookalike_agrupado['purchase_conversion_value'] / valor_total_compra) * 100
lookalike_agrupado = lookalike_agrupado.round(2)
lookalike_agrupado.head()

data = lookalike_agrupado
colors = {
    'Facebook': 'blue',
    'Instagram': 'orange',
}

fig_reach = go.Figure()
for platforma in data['platform'].unique():
    fig_reach.add_trace(go.Bar(
        x=[platforma],
        y=data[data['platform'] == platforma]['conversion_rate'],
        name='Taxa de conversão',
        marker=dict(color=colors.get(platforma, 'grey')),  
        showlegend=False  
    ))

fig_reach.update_layout(
    title='Taxa de conversão por plataforma',
    xaxis_title='Plataformas',
    yaxis_title='Taxa de conversão',
    width=600, 
    height=600, 
)

fig_reach.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=lookalike_agrupado, x='platform', y='reach', palette='viridis')
plt.title('Total de alcance por plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de alcance')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=lookalike_agrupado, x='platform', y='amount_spent_usd', palette='viridis')
plt.title('Total de compras por plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=lookalike_agrupado, x='platform', y='purchase', palette='viridis')
plt.title('Total de compras por plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('Total de compras')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=lookalike_agrupado, x='platform', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por Plataforma - Campanha "Lookalike Conversion"')
plt.ylabel('CPC (USD)')
plt.xlabel('Plataforma')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=lookalike_agrupado, x='reach', y='purchase', hue='platform', palette='viridis', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid()
plt.show()

dataframe_convertido_lookalike = plataforma_dataframe_ref[plataforma_dataframe_ref['group_campanha'] == 'Lookalike Conversion']

grouped_lookalike_simu = dataframe_convertido_lookalike.groupby(['platform', 'group_campanha']).agg({
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

print(plataforma_dataframe_ref.columns)

add_to_cart_df = plataforma_dataframe_ref[plataforma_dataframe_ref['group_campanha'] == 'Add to cart']

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

qtd_total_gasto = grouped_add_to_cart['amount_spent_usd'].sum()
valor_total_compra = grouped_add_to_cart['purchase_conversion_value'].sum()

grouped_add_to_cart['percentage_amount_spent_usd'] = (grouped_add_to_cart['amount_spent_usd'] / qtd_total_gasto) * 100
grouped_add_to_cart['percentage_purchase_conversion_value'] = (grouped_add_to_cart['purchase_conversion_value'] / valor_total_compra) * 100

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

viewed_df = plataforma_dataframe_ref[plataforma_dataframe_ref['group_campanha'] == 'Viewed']

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

qtd_total_gasto = grouped_viewed['amount_spent_usd'].sum()
valor_total_compra = grouped_viewed['purchase_conversion_value'].sum()

grouped_viewed['percentage_amount_spent_usd'] = (grouped_viewed['amount_spent_usd'] / qtd_total_gasto) * 100
grouped_viewed['percentage_purchase_conversion_value'] = (grouped_viewed['purchase_conversion_value'] / valor_total_compra) * 100

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

segmentation_df = plataforma_dataframe_ref[plataforma_dataframe_ref['group_campanha'] == 'Segmentação Demográfica e Geográfica']

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

qtd_total_gasto_segmentacao = grouped_segmentation['amount_spent_usd'].sum()
valor_total_compra_segmentacao = grouped_segmentation['purchase_conversion_value'].sum()

grouped_segmentation['percentage_amount_spent_usd'] = (grouped_segmentation['amount_spent_usd'] / qtd_total_gasto_segmentacao) * 100
grouped_segmentation['percentage_purchase_conversion_value'] = (grouped_segmentation['purchase_conversion_value'] / valor_total_compra_segmentacao) * 100

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

instagram_campaign_df = plataforma_dataframe_ref[plataforma_dataframe_ref['group_campanha'] == 'Instagram Campanha']

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

qtd_total_gasta_instagram = grouped_instagram['amount_spent_usd'].sum()
valor_total_compra_instagram = grouped_instagram['purchase_conversion_value'].sum()

grouped_instagram['percentage_amount_spent_usd'] = (grouped_instagram['amount_spent_usd'] / qtd_total_gasta_instagram) * 100
grouped_instagram['percentage_purchase_conversion_value'] = (grouped_instagram['purchase_conversion_value'] / valor_total_compra_instagram) * 100

grouped_instagram = grouped_instagram.round(2)

grouped_instagram.head()

plot_correlation_matrix(idade_dataframe_normal, selected_columns, title='Matriz de Correlação - BY_AGE')

group_gereal_age = idade_dataframe_ref.groupby('age').agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

group_gereal_age['ROI (%)'] = ((group_gereal_age['purchase_conversion_value'] - group_gereal_age['amount_spent_usd']) / group_gereal_age['amount_spent_usd']) * 100

qtd_total_gasto = group_gereal_age['amount_spent_usd'].sum()
valor_total_compra = group_gereal_age['purchase_conversion_value'].sum()

group_gereal_age['percentage_amount_spent_usd'] = (group_gereal_age['amount_spent_usd'] / qtd_total_gasto) * 100
group_gereal_age['percentage_purchase_conversion_value'] = (group_gereal_age['purchase_conversion_value'] / valor_total_compra) * 100

group_gereal_age = group_gereal_age[['age', 'amount_spent_usd', 'percentage_amount_spent_usd',
                                  'purchase_conversion_value', 'percentage_purchase_conversion_value',
                                  'ROI (%)']]
group_gereal_age.head()

group_gereal_age = idade_dataframe_ref.groupby(['age', 'group_campanha']).agg({
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

group_gereal_age['ROI (%)'] = ((group_gereal_age['purchase_conversion_value'] - group_gereal_age['amount_spent_usd']) / group_gereal_age['amount_spent_usd']) * 100

qtd_total_gasto = group_gereal_age['amount_spent_usd'].sum()
valor_total_compra = group_gereal_age['purchase_conversion_value'].sum()

group_gereal_age['percentage_amount_spent_usd'] = (group_gereal_age['amount_spent_usd'] / qtd_total_gasto) * 100
group_gereal_age['percentage_purchase_conversion_value'] = (group_gereal_age['purchase_conversion_value'] / valor_total_compra) * 100

group_gereal_age['ROI (%)'] = group_gereal_age['ROI (%)'].round(2)
group_gereal_age['percentage_amount_spent_usd'] = group_gereal_age['percentage_amount_spent_usd'].round(2)
group_gereal_age['percentage_purchase_conversion_value'] = group_gereal_age['percentage_purchase_conversion_value'].round(2)

group_gereal_age = group_gereal_age[['group_campanha', 'age', 'amount_spent_usd',
                   'percentage_amount_spent_usd','purchase_conversion_value',
                   'percentage_purchase_conversion_value','ROI (%)']]

group_gereal_age.head(37)

lookalike_conversion_age_df = idade_dataframe_ref[idade_dataframe_ref['group_campanha'] == 'Lookalike Conversion']

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

qtd_total_gasto = grouped_lookalike_age['amount_spent_usd'].sum()
valor_total_compra = grouped_lookalike_age['purchase_conversion_value'].sum()

grouped_lookalike_age['percentage_amount_spent_usd'] = (grouped_lookalike_age['amount_spent_usd'] / qtd_total_gasto) * 100
grouped_lookalike_age['percentage_purchase_conversion_value'] = (grouped_lookalike_age['purchase_conversion_value'] / valor_total_compra) * 100

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

idade_dataframe_add_to_cart = idade_dataframe_ref[idade_dataframe_ref['group_campanha'] == 'Add to cart']

agrupado_idade_add_to_cart = idade_dataframe_add_to_cart.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

agrupado_idade_add_to_cart['conversion_rate'] = (agrupado_idade_add_to_cart['purchase'] / agrupado_idade_add_to_cart['link_clicks']) * 100
agrupado_idade_add_to_cart['cpc_link_usd'] = agrupado_idade_add_to_cart['amount_spent_usd'] / agrupado_idade_add_to_cart['link_clicks']

agrupado_idade_add_to_cart['roi(%)'] = ((agrupado_idade_add_to_cart['purchase_conversion_value'] - agrupado_idade_add_to_cart['amount_spent_usd']) / agrupado_idade_add_to_cart['amount_spent_usd']) * 100

qtd_total_gasto = agrupado_idade_add_to_cart['amount_spent_usd'].sum()
valor_total_compra = agrupado_idade_add_to_cart['purchase_conversion_value'].sum()

agrupado_idade_add_to_cart['percentage_amount_spent_usd'] = (agrupado_idade_add_to_cart['amount_spent_usd'] / qtd_total_gasto) * 100
agrupado_idade_add_to_cart['percentage_purchase_conversion_value'] = (agrupado_idade_add_to_cart['purchase_conversion_value'] / valor_total_compra) * 100

agrupado_idade_add_to_cart = agrupado_idade_add_to_cart.round(2)
agrupado_idade_add_to_cart = agrupado_idade_add_to_cart[['age', 'group_campanha', 'reach', 'link_clicks', 'cpc_link_usd', 'purchase', 'conversion_rate','amount_spent_usd','percentage_amount_spent_usd', 'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)' ]]
agrupado_idade_add_to_cart.head()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=grouped_lookalike_age, x='cpc_link_usd', y='link_clicks', hue='age', palette='viridis', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=agrupado_idade_add_to_cart, x='age', y='link_clicks', palette='viridis')
plt.title('Total de clicks por idade - Campanha "Add to Cart"')
plt.ylabel('Total de clicks')
plt.xlabel('idade')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=agrupado_idade_add_to_cart, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Valor do click por idade - Campanha "Add to Cart"')
plt.ylabel('Valor click')
plt.xlabel('idade')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=agrupado_idade_add_to_cart, x='age', y='roi(%)', palette='viridis')
plt.title('Retorno sobre Investimento (ROI) por idade - Campanha "Add to Cart"')
plt.ylabel('ROI (%)')
plt.xlabel('idade')
plt.grid(axis='y')
plt.show()

idade_dataframe_viewed = idade_dataframe_ref[idade_dataframe_ref['group_campanha'] == 'Viewed']

viewed_agrupado_idade = idade_dataframe_viewed.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'  
}).reset_index()

viewed_agrupado_idade['ctr_all'] = (viewed_agrupado_idade['link_clicks'] / viewed_agrupado_idade['reach']) * 100
viewed_agrupado_idade['cpm_usd'] = (viewed_agrupado_idade['amount_spent_usd'] / viewed_agrupado_idade['reach']) * 1000
viewed_agrupado_idade['cpc_link_usd'] = viewed_agrupado_idade['amount_spent_usd'] / viewed_agrupado_idade['link_clicks']
viewed_agrupado_idade['roi'] = ((viewed_agrupado_idade['purchase_conversion_value'] - viewed_agrupado_idade['amount_spent_usd']) / viewed_agrupado_idade['amount_spent_usd']) * 100

qtd_total_gasto = viewed_agrupado_idade['amount_spent_usd'].sum()
valor_total_compra = viewed_agrupado_idade['purchase_conversion_value'].sum()

viewed_agrupado_idade['percentage_amount_spent_usd'] = (viewed_agrupado_idade['amount_spent_usd'] / qtd_total_gasto) * 100
viewed_agrupado_idade['percentage_purchase_conversion_value'] = (viewed_agrupado_idade['purchase_conversion_value'] / valor_total_compra) * 100

viewed_agrupado_idade = viewed_agrupado_idade.round(2)
viewed_agrupado_idade.head()

plt.figure(figsize=(10, 6))
sns.barplot(data=viewed_agrupado_idade, x='age', y='reach', palette='viridis')
plt.title('Alcance por idade - Campanha Viewed')
plt.xlabel('Idade')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=viewed_agrupado_idade, x='age', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por idade - Campanha Viewed')
plt.xlabel('idade')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=viewed_agrupado_idade, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por idade - Campanha Viewed')
plt.xlabel('Idade')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=viewed_agrupado_idade, x='age', y='roi', palette='viridis')
plt.title('ROI por idade - Campanha Viewed')
plt.xlabel('Idade')
plt.ylabel('Retorno sobre Investimento (%)')
plt.grid(axis='y')
plt.show()

idade_dataframe_segmentacao = idade_dataframe_ref[idade_dataframe_ref['group_campanha'] == 'Segmentação Demográfica e Geográfica']

agrupado_idade_segmentacao = idade_dataframe_segmentacao.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'mean',  
    'ctr_all': 'mean',     
    'cpm_usd': 'mean',     
    'cpc_link_usd': 'mean' 
}).reset_index()

agrupado_idade_segmentacao['roi'] = ((agrupado_idade_segmentacao['purchase_conversion_value'] - agrupado_idade_segmentacao['amount_spent_usd']) / agrupado_idade_segmentacao['amount_spent_usd']) * 100

qtd_total_gasto_segmentacao = agrupado_idade_segmentacao['amount_spent_usd'].sum()
valor_total_compra_segmentacao = agrupado_idade_segmentacao['purchase_conversion_value'].sum()

agrupado_idade_segmentacao['percentage_amount_spent_usd'] = (agrupado_idade_segmentacao['amount_spent_usd'] / qtd_total_gasto_segmentacao) * 100
agrupado_idade_segmentacao['percentage_purchase_conversion_value'] = (agrupado_idade_segmentacao['purchase_conversion_value'] / valor_total_compra_segmentacao) * 100

agrupado_idade_segmentacao = agrupado_idade_segmentacao.round(2)
agrupado_idade_segmentacao.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_segmentacao, x='age', y='amount_spent_usd', palette='viridis')
plt.title('Amount spent - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Amount spent')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_segmentacao, x='age', y='frequency', palette='viridis')
plt.title('Frequência Média por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Frequência Média')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_segmentacao, x='age', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_segmentacao, x='age', y='cpm_usd', palette='viridis')
plt.title('Custo médio por mil impressões - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Custo médio')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_segmentacao, x='age', y='reach', palette='viridis')
plt.title('Alcance por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_segmentacao, x='age', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por Idade - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

idade_dataframe_instagram = idade_dataframe_ref[idade_dataframe_ref['group_campanha'] == 'Instagram Campanha']

agrupado_idade_instagram = idade_dataframe_instagram.groupby(['age', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum',
    'frequency': 'sum',  
    'ctr_all': 'mean',   
    'cpm_usd': 'mean',   
    'cpc_link_usd': 'mean'  
}).reset_index()

agrupado_idade_instagram['roi'] = ((agrupado_idade_instagram['purchase_conversion_value'] - agrupado_idade_instagram['amount_spent_usd']) / agrupado_idade_instagram['amount_spent_usd']) * 100

qtd_total_gasta_instagram = agrupado_idade_instagram['amount_spent_usd'].sum()
valor_total_compra_instagram = agrupado_idade_instagram['purchase_conversion_value'].sum()

agrupado_idade_instagram['percentage_amount_spent_usd'] = (agrupado_idade_instagram['amount_spent_usd'] / qtd_total_gasta_instagram) * 100
agrupado_idade_instagram['percentage_purchase_conversion_value'] = (agrupado_idade_instagram['purchase_conversion_value'] / valor_total_compra_instagram) * 100

agrupado_idade_instagram = agrupado_idade_instagram.round(2)
agrupado_idade_instagram.head()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_instagram, x='age', y='reach', palette='viridis')
plt.title('reach - Instagram')
plt.xlabel('Idade')
plt.ylabel('Custo médio')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_idade_instagram, x='age', y='cpm_usd', palette='viridis')
plt.title('Alcance - Instagram')
plt.xlabel('Idade')
plt.ylabel('Alcance')
plt.grid(axis='y')
plt.show()

plot_correlation_matrix(pais_dataframe_normal, selected_columns, title='Matriz de Correlação - BY_COUNTRY')

print(pais_dataframe_ref.columns)

pais_dataframe_ref.head(20)

pais_dataframe_conversao_lookalike = pais_dataframe_ref[pais_dataframe_ref['group_campanha'] == 'Lookalike Conversion']

agrupado_pais_lookalike = pais_dataframe_conversao_lookalike.groupby('country_name').agg({
    'result_rate': 'sum',
    'population': 'mean',
    'reach': 'sum',
    'link_clicks': 'sum',
    'purchase': 'sum',
    'amount_spent_usd': 'sum',
    'purchase_conversion_value': 'sum'
}).reset_index()

agrupado_pais_lookalike['conversion_rate (%)'] = (agrupado_pais_lookalike['purchase'] / agrupado_pais_lookalike['link_clicks']) * 100
agrupado_pais_lookalike['cpc_link_usd'] = agrupado_pais_lookalike['amount_spent_usd'] / agrupado_pais_lookalike['link_clicks']
agrupado_pais_lookalike['cpa_usd'] = agrupado_pais_lookalike['amount_spent_usd'] / agrupado_pais_lookalike['purchase']
agrupado_pais_lookalike['reach_per_population (%)'] = (agrupado_pais_lookalike['reach'] / agrupado_pais_lookalike['population']) * 100
agrupado_pais_lookalike['ROI (%)'] = ((agrupado_pais_lookalike['purchase_conversion_value'] - agrupado_pais_lookalike['amount_spent_usd']) / agrupado_pais_lookalike['amount_spent_usd']) * 100

qtd_total_gasto = agrupado_pais_lookalike['amount_spent_usd'].sum()
valor_total_compra = agrupado_pais_lookalike['purchase_conversion_value'].sum()
agrupado_pais_lookalike['percentage_amount_spent_usd'] = (agrupado_pais_lookalike['amount_spent_usd'] / qtd_total_gasto) * 100
agrupado_pais_lookalike['percentage_purchase_conversion_value'] = (agrupado_pais_lookalike['purchase_conversion_value'] / valor_total_compra) * 100

agrupado_pais_lookalike = agrupado_pais_lookalike.loc[agrupado_pais_lookalike['ROI (%)'] > -100.00]

agrupado_pais_lookalike = agrupado_pais_lookalike.dropna()

colunas_finais = [
    'country_name', 'population', 'reach_per_population (%)', 'result_rate', 'conversion_rate (%)', 'link_clicks', 'cpc_link_usd',
    'cpa_usd', 'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'ROI (%)'
]
agrupado_pais_lookalike = agrupado_pais_lookalike[colunas_finais].round(2).sort_values(by='reach_per_population (%)', ascending=False).reset_index(drop=True)

agrupado_pais_lookalike.head(10)

top_10 = agrupado_pais_lookalike.head(10)

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

add_to_cart_country_df = pais_dataframe_ref[pais_dataframe_ref['group_campanha'] == 'Add to cart']

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

qtd_total_gasto = grouped_add_to_cart_country['amount_spent_usd'].sum()
valor_total_compra = grouped_add_to_cart_country['purchase_conversion_value'].sum()

grouped_add_to_cart_country['percentage_amount_spent_usd'] = (grouped_add_to_cart_country['amount_spent_usd'] / qtd_total_gasto) * 100
grouped_add_to_cart_country['percentage_purchase_conversion_value'] = (grouped_add_to_cart_country['purchase_conversion_value'] / valor_total_compra) * 100

grouped_add_to_cart_country = grouped_add_to_cart_country.loc[grouped_add_to_cart_country['roi(%)'] > -100.00]

grouped_add_to_cart_country = grouped_add_to_cart_country.dropna()

colunas_finais = [
    'country_name', 'population', 'reach_per_population (%)', 'conversion_rate', 'link_clicks', 'cpc_link_usd',
    'purchase', 'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)'
]

grouped_add_to_cart_country = grouped_add_to_cart_country[colunas_finais].round(2).sort_values(
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

pais_dataframe_viewed = pais_dataframe_ref[pais_dataframe_ref['group_campanha'] == 'Viewed']

agrupado_pais_viewed = pais_dataframe_viewed.groupby(['country_name', 'group_campanha']).agg({
    'reach': 'sum',
    'link_clicks': 'sum',
    'amount_spent_usd': 'sum',
    'population': 'mean',
    'purchase_conversion_value': 'sum',
    'purchase': 'sum'
}).reset_index()

agrupado_pais_viewed['ctr_all'] = (agrupado_pais_viewed['link_clicks'] / agrupado_pais_viewed['reach']) * 100
agrupado_pais_viewed['cpm_usd'] = (agrupado_pais_viewed['amount_spent_usd'] / agrupado_pais_viewed['reach']) * 1000
agrupado_pais_viewed['reach_per_population (%)'] = (agrupado_pais_viewed['reach'] / agrupado_pais_viewed['population']) * 100
agrupado_pais_viewed['cpc_link_usd'] = agrupado_pais_viewed['amount_spent_usd'] / agrupado_pais_viewed['link_clicks']
agrupado_pais_viewed['roi(%)'] = ((agrupado_pais_viewed['purchase_conversion_value'] - agrupado_pais_viewed['amount_spent_usd']) / agrupado_pais_viewed['amount_spent_usd']) * 100
agrupado_pais_viewed['conversion_rate'] = (agrupado_pais_viewed['purchase'] / agrupado_pais_viewed['link_clicks']) * 100

qtd_total_gasto = agrupado_pais_viewed['amount_spent_usd'].sum()
valor_total_compra = agrupado_pais_viewed['purchase_conversion_value'].sum()

agrupado_pais_viewed['percentage_amount_spent_usd'] = (agrupado_pais_viewed['amount_spent_usd'] / qtd_total_gasto) * 100
agrupado_pais_viewed['percentage_purchase_conversion_value'] = (agrupado_pais_viewed['purchase_conversion_value'] / valor_total_compra) * 100

agrupado_pais_viewed = agrupado_pais_viewed.loc[agrupado_pais_viewed['roi(%)'] > -100.00]
agrupado_pais_viewed = agrupado_pais_viewed.dropna()

colunas_finais = [
    'country_name', 'population', 'reach_per_population (%)', 'conversion_rate', 'ctr_all', 'link_clicks', 'cpc_link_usd', 'cpm_usd',
    'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)'
]

agrupado_pais_viewed = agrupado_pais_viewed[colunas_finais].round(2).sort_values(
    by=['reach_per_population (%)'], ascending=[False]  
).reset_index(drop=True)

agrupado_pais_viewed.head(10)

top_10 = agrupado_pais_viewed.head(10)

plt.figure(figsize=(15, 8))
sns.scatterplot(data=top_10, x='purchase_conversion_value', y='reach_per_population (%)', hue='country_name', palette='plasma', s=200)
plt.title('Relação entre Alcance e Compras - Campanha "Lookalike Conversion"')
plt.ylabel('Total de Compras')
plt.xlabel('Alcance')
plt.grid(axis='y')
plt.show()

top_10_viewed = agrupado_pais_viewed.head(10)

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

pais_dataframe_segmentacao = pais_dataframe_ref[pais_dataframe_ref['group_campanha'] == 'Segmentação Demográfica e Geográfica']

agrupado_pais_segmentacao = pais_dataframe_segmentacao.groupby(['country_name', 'group_campanha']).agg({
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


agrupado_pais_segmentacao['roi(%)'] = ((agrupado_pais_segmentacao['purchase_conversion_value'] - agrupado_pais_segmentacao['amount_spent_usd']) / agrupado_pais_segmentacao['amount_spent_usd']) * 100
agrupado_pais_segmentacao['reach_per_population (%)'] = (agrupado_pais_segmentacao['reach'] / agrupado_pais_segmentacao['population']) * 100
qtd_total_gasto_segmentacao = agrupado_pais_segmentacao['amount_spent_usd'].sum()
valor_total_compra_segmentacao = agrupado_pais_segmentacao['purchase_conversion_value'].sum()
agrupado_pais_segmentacao['percentage_amount_spent_usd'] = (agrupado_pais_segmentacao['amount_spent_usd'] / qtd_total_gasto_segmentacao) * 100
agrupado_pais_segmentacao['percentage_purchase_conversion_value'] = (agrupado_pais_segmentacao['purchase_conversion_value'] / valor_total_compra_segmentacao) * 100
agrupado_pais_segmentacao = agrupado_pais_segmentacao.loc[agrupado_pais_segmentacao['roi(%)'] > -100.00]
agrupado_pais_segmentacao = agrupado_pais_segmentacao.dropna()

colunas_finais = [
    'country_name', 'population', 'frequency', 'reach_per_population (%)', 'link_clicks', 'cpc_link_usd', 'cpm_usd',
    'ctr_all', 'amount_spent_usd', 'percentage_amount_spent_usd',
    'purchase_conversion_value', 'percentage_purchase_conversion_value', 'roi(%)'
]

agrupado_pais_segmentacao = agrupado_pais_segmentacao[colunas_finais].round(2).sort_values(
    by=['reach_per_population (%)'], ascending=[False]  
).reset_index(drop=True)

agrupado_pais_segmentacao.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_pais_segmentacao, x='country_name', y='amount_spent_usd', palette='viridis')
plt.title('Amount spent - Segmentação Demográfica e Geográfica')
plt.xlabel('Idade')
plt.ylabel('Amount spent')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_pais_segmentacao, x='country_name', y='reach_per_population (%)', palette='viridis')
plt.title('Alcande por população(%) - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Alcande por população(%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_pais_segmentacao, x='country_name', y='frequency', palette='viridis')
plt.title('Frequência Média por País - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Frequência Média')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_pais_segmentacao, x='country_name', y='ctr_all', palette='viridis')
plt.title('Taxa de Cliques (CTR) por País - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Taxa de Cliques (%)')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_pais_segmentacao, x='country_name', y='cpm_usd', palette='viridis')
plt.title('Custo médio por mil impressões - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Custo médio')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=agrupado_pais_segmentacao, x='country_name', y='cpc_link_usd', palette='viridis')
plt.title('Custo por Clique (CPC) por País - Segmentação Demográfica e Geográfica')
plt.xlabel('País')
plt.ylabel('Custo por Clique (USD)')
plt.grid(axis='y')
plt.show()

pais_dataframe_instagram = pais_dataframe_ref[pais_dataframe_ref['group_campanha'] == 'Instagram Campanha']

pais_agrupado_instagram = pais_dataframe_instagram.groupby(['country_name', 'group_campanha']).agg({
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

pais_agrupado_instagram['roi(%)'] = ((pais_agrupado_instagram['purchase_conversion_value'] - pais_agrupado_instagram['amount_spent_usd']) / pais_agrupado_instagram['amount_spent_usd']) * 100

qtd_total_gasta_instagram = pais_agrupado_instagram['amount_spent_usd'].sum()
valor_total_compra_instagram = pais_agrupado_instagram['purchase_conversion_value'].sum()

pais_agrupado_instagram['percentage_amount_spent_usd'] = (pais_agrupado_instagram['amount_spent_usd'] / qtd_total_gasta_instagram) * 100
pais_agrupado_instagram['percentage_purchase_conversion_value'] = (pais_agrupado_instagram['purchase_conversion_value'] / valor_total_compra_instagram) * 100

pais_agrupado_instagram = pais_agrupado_instagram.round(2)
pais_agrupado_instagram.head()

text = " ".join(idade_dataframe['Ad Set Name'].dropna().astype(str))

incidencia_idade = idade_dataframe_ref['group_campanha'].value_counts()
incidencia_pais = pais_dataframe_ref['group_campanha'].value_counts()
incidencia_plataforma = plataforma_dataframe_ref['group_campanha'].value_counts()

incidencia_total = pd.DataFrame({
    'group_campanha': incidencia_idade.index,
    'age_count': incidencia_idade.values,
    'country_count': incidencia_pais.reindex(incidencia_idade.index, fill_value=0).values,
    'platform_count': incidencia_plataforma.reindex(incidencia_idade.index, fill_value=0).values
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
   title='Distribuição média das campanhas',
    title_font=dict(size=20, color='black'),
    xaxis_title='Tipo de campanha',
    yaxis_title='Número de campanhas',
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

fig = px.treemap(df, path=[px.Constant("Mundo"), 'continent', 'country'],values='investment', color='investment',hover_data=['iso_alpha'],color_continuous_scale='RdBu',color_continuous_midpoint=df['investment'].mean())
fig.update_traces(texttemplate="%{label}: %{value}", textfont_size=12)
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),title='Investimentos por País',)

fig.show()

data = {
    'plataforma': ['Instagram', 'Facebook'],
    'investment': [3000, 7000],
}

df = pd.DataFrame(data)

df['categoria'] = 'Plataforma'
df['iso_alpha'] = ['INS', 'FB'] 

fig = px.treemap(df, path=[px.Constant("Internet"), 'categoria', 'plataforma'],values='investment', color='investment',hover_data=['iso_alpha'],color_continuous_scale='RdBu',color_continuous_midpoint=df['investment'].mean())
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

fig = px.treemap(df, path=[px.Constant("Mundo"), 'categoria', 'faixa_etaria'],values='investment', color='investment',hover_data=['iso_alpha'],color_continuous_scale='RdBu',color_continuous_midpoint=df['investment'].mean())
fig.update_traces(texttemplate="%{label}: %{value}", textfont_size=12)
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),title='Investimentos por faixa etária')

fig.show()