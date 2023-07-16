import csv
import glob
import os
import pandas as pd


def format_txt_to_csv():
    folder_path = "data/txt_format/"
    output_path = "data/csv_format/"

    txt_files = glob.glob(folder_path + "*.txt")

    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        new_file_name = os.path.splitext(file_name)[0] + ".csv"
        with open(file_path, 'r') as file_input:
            lines = file_input.readlines()
            lines = [line.strip() for line in lines]
            data = [line.split('\t') for line in lines]

        with open(output_path + os.path.basename(new_file_name), 'w', newline='') as file_out:
            writer = csv.writer(file_out, delimiter=';')
            writer.writerows(data)
        print("Sucessfully converted ", file_name)

def group_data(split_date=2021):
    folder_path = "data/csv_format/"
    futur_path = "data/futur"
    past_path = "data/past"

    datas_past = pd.DataFrame([])
    datas_futur = pd.DataFrame([])

    csv_files = glob.glob(folder_path + "*.csv")
    for file_path in csv_files:
        data = pd.read_csv(file_path, sep=";")
        data["NB_VALD"] = data["NB_VALD"].replace('Moins de 5', 1)

        year = int(os.path.basename(file_path)[0:4])
        if year >= split_date:
            datas_futur = pd.concat([datas_futur, data])
        else:
            datas_past = pd.concat([datas_past, data])

    datas_past.to_csv(past_path + "/validation_past.csv", sep=";", index=False, header=True)
    datas_futur.to_csv(futur_path + "/validation_futur.csv", sep=";", index=False, header=True)

def groupby_date_and_save(data_path, save_path, group_list):
    data = pd.read_csv(data_path, sep=";")
    data['JOUR'] = pd.to_datetime(data['JOUR'], format='%d/%m/%Y').dt.date
    result = data.groupby(group_list)['NB_VALD'].sum().reset_index()
    result.to_csv(save_path + "validation_groupby_"+"_".join(group_list)+".csv", sep=";", index=False, header=True)


# format_txt_to_csv()
# group_data()
# groupby_date_and_save("data/futur/validation_futur.csv", "data/futur/", ['JOUR','CATEGORIE_TITRE'])
# groupby_date_and_save("data/past/validation_past.csv", "data/past/", ['JOUR'])
# groupby_date_and_save("data/past/validation_past.csv", "data/past/", ['JOUR','LIBELLE_ARRET'])
# groupby_date_and_save("data/past/validation_past.csv", "data/past/", ['JOUR','CATEGORIE_TITRE'])












