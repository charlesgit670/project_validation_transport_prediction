import os
import pandas as pd
import glob


def read_most_recent_csv(directory_path, number_files=46):
    file_pattern = os.path.join(directory_path, "*csv")
    files = glob.glob(file_pattern)

    # Trier les fichiers par date de modification (du plus récent au plus ancien)
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(x))
    latest_files = sorted_files[-number_files:]

    # Lire et agréger les données des fichiers
    dataframes = []
    for file_path in latest_files:
        df = pd.read_csv(file_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
        dataframes.append(df)
    merged_df = pd.concat(dataframes)

    return merged_df