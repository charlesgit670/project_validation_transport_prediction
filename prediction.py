import os
import numpy as np

from joblib import load
from keras.models import load_model


def predict_total_validation(df):
    window_size = 45
    assert(len(df) >= window_size)

    scaler_dir = 'model/scaler'
    model_dir = 'model'
    scaler = load(os.path.join(scaler_dir, "scaler_total_validation.joblib"))

    data = scaler.transform(df['NB_VALD'].values.reshape(-1, 1))
    data = data[-window_size:,:]
    model = load_model(os.path.join(model_dir, 'lstm_total_validation.h5'))
    predicted_value = model.predict(data.reshape(1,-1,1))
    predicted_value = scaler.inverse_transform(predicted_value).reshape(-1)

    return predicted_value


def predict_titre_validation(df):
    window_size = 45
    scaler_dir = 'model/scaler'
    model_dir = 'model'

    data = []
    scaler_dict = {}

    unique_titre = np.load("model/CATEGORIE_TITRE.npy")
    for titre in unique_titre:
        df_filter = df[df["CATEGORIE_TITRE"] == titre]
        assert (len(df_filter) >= 45)
        scaler_dict[titre] = load(os.path.join(scaler_dir, "scaler_titre_"+titre+".joblib"))
        data_tmp = scaler_dict[titre].transform(df_filter['NB_VALD'].values.reshape(-1, 1))
        data_tmp = data_tmp[-window_size:, :]
        data.append(np.expand_dims(data_tmp, axis=0))

    data = np.concatenate(data, axis=2)

    model = load_model(os.path.join(model_dir, 'lstm_titre_validation.h5'))
    predicted_value = model.predict(data).reshape(-1)

    for i, titre in enumerate(unique_titre):
        predicted_value[i] = scaler_dict[titre].inverse_transform(predicted_value[i].reshape(-1, 1)).reshape(1)

    return predicted_value


def predict_arret_validation(df):
    window_size = 45
    scaler_dir = 'model/scaler'
    model_dir = 'model'

    data = []
    scaler_dict = {}

    unique_arret = np.load("model/LIBELLE_ARRET.npy")
    for arret in unique_arret:
        df_filter = df[df["LIBELLE_ARRET"] == arret]
        assert (len(df_filter) >= 45)
        scaler_dict[arret] = load(os.path.join(scaler_dir, "scaler_titre_" + arret + ".joblib"))
        data_tmp = scaler_dict[arret].transform(df_filter['NB_VALD'].values.reshape(-1, 1))
        data_tmp = data_tmp[-window_size:, :]
        data.append(np.expand_dims(data_tmp, axis=0))

    data = np.concatenate(data, axis=2)

    model = load_model(os.path.join(model_dir, 'lstm_arret_validation.h5'))
    predicted_value = model.predict(data).reshape(-1)

    for i, arret in enumerate(unique_arret):
        predicted_value[i] = scaler_dict[arret].inverse_transform(predicted_value[i].reshape(-1, 1)).reshape(1)

    return predicted_value