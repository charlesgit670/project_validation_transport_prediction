from read_data_stream import read_most_recent_csv
from main_visualisation import monitoring_today_validation

if __name__ == "__main__":
    directory_total_path = ""
    directory_titre_path = ""
    directory_arret_path = ""

    df_total = read_most_recent_csv(directory_total_path)
    df_titre = read_most_recent_csv(directory_titre_path)
    df_arret = read_most_recent_csv(directory_arret_path)

    monitoring_today_validation(df_total, df_titre, df_arret)
