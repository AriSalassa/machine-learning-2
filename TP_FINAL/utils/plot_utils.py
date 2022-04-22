import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


def display_dataset_distributions(dataset):
    fig = dataset.hist(xlabelsize=12, ylabelsize=12, figsize=(22, 10))
    [x.title.set_size(14) for x in fig.ravel()]
    plt.tight_layout()
    plt.show()


def display_dataset_bar_value_count(dataset):
    # Columnas del dataset
    columns = dataset.columns
    n_columns = len(columns)

    # Número de columnas en la imagen
    plot_columns = 3
    plot_rows = int(n_columns/plot_columns) if n_columns%plot_columns == 0 else int(n_columns/plot_columns) + 1
    

    fig, ax = plt.subplots(plot_rows, plot_columns, figsize=(12,6))
    if ax.ndim == 1:
        ax = ax.reshape((plot_rows, plot_columns))

    for i, column in enumerate(columns):
        row = int(i / plot_columns)
        col = i % plot_columns
        dataset[column].value_counts().plot(kind="bar", ax=ax[row, col]).set_title(column)
        ax[row, col].grid()

    plt.tight_layout()
    fig.show()

def plot_cat_var_relations(X_in, y_in, label_name, column_names=None):
    if column_names is not None:
        df_temp = X_in[column_names].join(y_in).copy()
    else:
        df_temp = X_in.join(y_in).copy()
        column_names = X_in.columns
    
    # Número de columnas en la imagen
    plot_columns = 3
    plot_rows = int(len(column_names)/plot_columns) if len(column_names)%plot_columns == 0 else int(len(column_names)/plot_columns) + 1

    fig, ax = plt.subplots(plot_rows, plot_columns, figsize=(12,6))
    if ax.ndim == 1:
        ax = ax.reshape((plot_rows, plot_columns))

    fig.suptitle(f'Monotonic relationship with target variable: {label_name}')

    for i, column in enumerate(column_names):
        row = int(i / plot_columns)
        col = i % plot_columns

        ordered_labels = df_temp.groupby([column])[label_name].mean().sort_values().index
        aux_d = {k: i for i, k in enumerate(ordered_labels, 0)}
        
        df_temp[column] = df_temp[column].map(aux_d)
        
        df_temp[[column, label_name]].groupby([column])[label_name].mean().plot(ax=ax[row, col])

        ax[row, col].set_title(f'{column}\n(encoded)')
        ax[row, col].set_ylabel(f'{label_name} (average)')
        ax[row, col].grid()
    
    plt.tight_layout()
    
    return

def predict_and_plot(preds, targets, name=''):    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    sns.heatmap(cf, ax=axes[0], annot=True)
    axes[0].set_xlabel('Prediction')
    axes[0].set_ylabel('Target')
    axes[0].set_title('{} Confusion Matrix -Normalized-'.format(name))

    cf = confusion_matrix(targets, preds)

    sns.heatmap(cf, ax=axes[1], annot=True,)
    axes[1].set_xlabel('Prediction')
    axes[1].set_ylabel('Target')
    axes[1].set_title('{} Confusion Matrix'.format(name))

    plt.tight_layout()
