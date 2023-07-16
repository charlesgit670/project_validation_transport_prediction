import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prediction import predict_arret_validation, predict_titre_validation, predict_total_validation

def plot_histogramme(ax, actual_values, predicted_values, x_labels, real_variations, xlabel, titre):
    indices = np.arange(len(x_labels))
    largeur_barre = 0.4

    ax.bar(indices, actual_values, width=largeur_barre, color='blue', label='Valeurs Actuelles')
    ax.bar(indices + largeur_barre, predicted_values, width=largeur_barre, color='orange', label='Valeurs Prédites')

    # Ajouter la variation au-dessus des barres
    for i, real_variation in enumerate(real_variations):
        ax.text(indices[i] + largeur_barre / 2, max(actual_values[i], predicted_values[i]), f'{round(real_variation*100)}%',
                 ha='center', va='bottom')

    # Configurer les étiquettes de l'axe x
    ax.set_xticks(indices + largeur_barre / 2)
    ax.set_xticklabels(x_labels, fontsize=8)

    # Titres des axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Nombre de validation')
    ax.set_title(titre)

    ax.legend()

def monitoring_today_validation(df_total, df_titre, df_arret):

    current_day = df_total.index.max()

    unique_titre = np.load("model/CATEGORIE_TITRE.npy")
    unique_arret = np.load("model/LIBELLE_ARRET.npy")

    actual_total_validation = df_total[df_total.index == current_day]
    actual_titre_validation = df_titre[df_titre.index == current_day]
    actual_arret_validation = df_arret[df_arret.index == current_day]

    actual_titre_validation = actual_titre_validation[actual_titre_validation["CATEGORIE_TITRE"].isin(unique_titre)]
    actual_arret_validation = actual_arret_validation[actual_arret_validation["LIBELLE_ARRET"].isin(unique_arret)]

    predicted_value_total = predict_total_validation(df_total[df_total.index < current_day])
    predicted_value_titre = predict_titre_validation(df_titre[df_titre.index < current_day])
    predicted_value_arret = predict_arret_validation(df_arret[df_arret.index < current_day])

    incertitude_total = np.load("model/incertitudes_total.npy")
    incertitude_titre = np.load("model/incertitudes_titre.npy")
    incertitude_arret = np.load("model/incertitudes_arret.npy")

    actual_total_variation = (actual_total_validation["NB_VALD"] - predicted_value_total)/actual_total_validation["NB_VALD"]
    actual_titre_variation = (actual_titre_validation["NB_VALD"] - predicted_value_titre)/actual_titre_validation["NB_VALD"]
    actual_arret_variation = (actual_arret_validation["NB_VALD"] - predicted_value_arret)/actual_arret_validation["NB_VALD"]

    actual_total_variation_minus_incertitude = np.maximum(abs(actual_total_variation) - incertitude_total, 0)
    actual_titre_variation_minus_incertitude = np.maximum(abs(actual_titre_variation) - incertitude_titre, 0)
    actual_arret_variation_minus_incertitude = np.maximum(abs(actual_arret_variation) - incertitude_arret, 0)

    indices_top5_largest_variation_arret = np.argsort(actual_arret_variation_minus_incertitude)[-5:]

    fig = plt.figure(figsize=(13, 10))

    # Définition de la grille personnalisée
    grid = (2, 3)

    ax1 = plt.subplot2grid(grid, (0, 0), colspan=3)
    ax2 = plt.subplot2grid(grid, (1, 0), colspan=1)
    ax3 = plt.subplot2grid(grid, (1, 1), colspan=2)

    # plot total
    plot_histogramme(ax2, actual_total_validation["NB_VALD"].tolist(), predicted_value_total, [''],
                     actual_total_variation_minus_incertitude, "Total", f"Nombre de validation le {current_day.strftime('%Y-%m-%d')}")
    # plot titre
    plot_histogramme(ax3, actual_titre_validation["NB_VALD"].tolist(), predicted_value_titre, unique_titre,
                     actual_titre_variation_minus_incertitude, "Titre de transport", f"Nombre de validation le {current_day.strftime('%Y-%m-%d')}")
    # plot arret
    plot_histogramme(ax1, np.array(actual_arret_validation["NB_VALD"])[indices_top5_largest_variation_arret],
                     predicted_value_arret[indices_top5_largest_variation_arret],
                     unique_arret[indices_top5_largest_variation_arret], np.array(actual_arret_variation_minus_incertitude)[indices_top5_largest_variation_arret],
                     "Nom de l'arrêt", f"Nombre de validation le {current_day.strftime('%Y-%m-%d')}")

    plt.subplots_adjust(hspace=0.15, wspace=0.15, left=0.07, right=0.95, bottom=0.05, top=0.95)
    plt.show()

def read_data_test():
    df_total = pd.read_csv("data/past/validation_groupby_JOUR.csv", sep=";", parse_dates=['JOUR'], index_col='JOUR')
    df_titre = pd.read_csv("data/past/validation_groupby_JOUR_CATEGORIE_TITRE.csv", sep=";", parse_dates=['JOUR'],index_col='JOUR')
    df_arret = pd.read_csv("data/past/validation_groupby_JOUR_LIBELLE_ARRET.csv", sep=";", parse_dates=['JOUR'],index_col='JOUR')

    df_total = df_total[(df_total.index >= "2019-01-01") & (df_total.index <= "2019-11-20")]
    df_titre = df_titre[(df_titre.index >= "2019-01-01") & (df_titre.index <= "2019-11-20")]
    df_arret = df_arret[(df_arret.index >= "2019-01-01") & (df_arret.index <= "2019-11-20")]

    monitoring_today_validation(df_total, df_titre, df_arret)


read_data_test()
