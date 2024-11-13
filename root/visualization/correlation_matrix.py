import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, selected_columns, title='Matriz de Correlação'):
    correlation_matrix = df[selected_columns].corr()

    print(correlation_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(title)
    plt.show()