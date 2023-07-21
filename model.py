import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.seasonal import seasonal_decompose
# from pmdarima.arima import auto_arima
import os
import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
# from keras_tuner.tuners import RandomSearch
# from tensorflow.keras.optimizers import Adam

# def plot_fft(data_path):
#     data = pd.read_csv(data_path, sep=";")
#     data['JOUR'] = pd.to_datetime(data['JOUR'], format='%Y-%m-%d')
#     data = data[data['JOUR'] <= "2019-12-31"]
#     x = (data['JOUR'] - data['JOUR'].iloc[0]).dt.days.values
#     y = data['NB_VALD'].values
#     fft_result = np.fft.fft(y)
#     freqs = np.arange(len(y))
#     plt.stem(freqs, np.abs(fft_result), 'b', \
#              markerfmt=" ", basefmt="-b")
#     plt.xlabel('Fréquence')
#     plt.ylabel('Amplitude')
#     plt.title('Analyse de Fourier')
#     plt.show()

def plot_data(data_path_past, data_path_futur):
    data_past = pd.read_csv(data_path_past, sep=";", parse_dates=['JOUR'], index_col='JOUR')
    data_futur = pd.read_csv(data_path_futur, sep=";", parse_dates=['JOUR'], index_col='JOUR')
    data = pd.concat([data_past, data_futur])
    data.plot()
    plt.show()

# def data_seasonal_decompose(data_path):
#     data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
#     data = data[data.index <= "2019-12-01"]
#     decomposition = seasonal_decompose(data, period=52)
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid
#
#     # Visualisation des composantes
#     plt.subplot(411)
#     plt.plot(data, label='Original')
#     plt.legend(loc='best')
#     plt.subplot(412)
#     plt.plot(trend, label='Trend')
#     plt.legend(loc='best')
#     plt.subplot(413)
#     plt.plot(seasonal, label='Seasonality')
#     plt.legend(loc='best')
#     plt.subplot(414)
#     plt.plot(residual, label='Residuals')
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()

# def arima_hyperparameter(data_path):
#     data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
#     model = auto_arima(data, seasonal=True, m=7)  # m est la période saisonnière
#     print(model.summary())

# def sarimax(data_path):
#     data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
#     data = data[data.index <= "2019-12-01"]
#
#     start_date = data.index[-1] + pd.DateOffset(days=1)
#     end_date = start_date + pd.DateOffset(days=365)
#
#     # Décomposition saisonnière hebdomadaire
#     decomposition_weekly = seasonal_decompose(data, period=7)
#     seasonal_component_weekly = decomposition_weekly.seasonal
#
#     # Décomposition saisonnière annuelle
#     decomposition_yearly = seasonal_decompose(data, period=365)
#     seasonal_component_yearly = decomposition_yearly.seasonal
#
#     # Modélisation SARIMA pour la composante saisonnière hebdomadaire
#     model_weekly = SARIMAX(seasonal_component_weekly, order=(4, 1, 2), seasonal_order=(2, 0, 2, 7))
#     sarima_weekly = model_weekly.fit()
#
#     # Modélisation SARIMA pour la composante saisonnière annuelle
#     model_yearly = SARIMAX(seasonal_component_yearly, order=(4, 1, 2), seasonal_order=(2, 0, 2, 365))
#     sarima_yearly = model_yearly.fit()
#
#     # Prédictions pour la composante saisonnière hebdomadaire
#     predicted_weekly = sarima_weekly.predict(start=start_date, end=end_date)
#
#     # Prédictions pour la composante saisonnière annuelle
#     predicted_yearly = sarima_yearly.predict(start=start_date, end=end_date)
#
#     # Prédiction finale en ajoutant les composantes saisonnières aux tendances
#     trend_component = decomposition_weekly.trend + decomposition_yearly.trend
#     predicted = trend_component + predicted_weekly + predicted_yearly
#
#     # Trace des prédictions et des données réelles
#     plt.plot(data.index, data, label='Données réelles')
#     plt.plot(predicted.index, predicted, label='Prédictions')
#     plt.legend()
#     plt.show()

def find_and_save_complete_name(data_path, column_name):
    data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
    data = data[data.index <= "2019-12-01"]
    save_path = 'model/'+column_name+'.npy'
    total_days = len(data.index.unique())

    keep_name = []

    unique_name = data[column_name].unique()
    for name in unique_name:
        data_filter = data[data[column_name] == name]
        if len(data_filter) == total_days:
            keep_name.append(name)

    np.save(save_path, np.array(keep_name))

# Création des ensembles d'entraînement et de test
def create_dataset(dataset, window_size):
    X, Y = [], []
    for i in range(len(dataset) - window_size):
        window = dataset[i:(i + window_size), 0]
        X.append(window)
        Y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(Y)

def model_total_validation(data_path, is_model_load=True, plot=False):
    data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
    data = data[data.index <= "2019-12-01"]
    data['jour_semaine'] = data.index.strftime('%w').astype(int)
    france_holidays = holidays.France()
    data['jour_ferie'] = data.index.to_series().apply(lambda d: d in france_holidays).astype(int)
    data_past = data[data.index <= "2018-12-31"]
    data_futur = data[(data.index > "2018-12-31")]

    # Prétraitement des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data_past['NB_VALD'].values.reshape(-1, 1))
    # Créer un dossier pour sauvegarder le scaler
    scaler_dir = 'model/scaler'
    os.makedirs(scaler_dir, exist_ok=True)
    dump(scaler, os.path.join(scaler_dir, "scaler_total_validation.joblib"))

    train_data = scaler.transform(data_past['NB_VALD'].values.reshape(-1, 1))
    test_data = scaler.transform(data_futur['NB_VALD'].values.reshape(-1, 1))

    window_size = 45
    train_X, train_Y = create_dataset(train_data, window_size)
    test_X, test_Y = create_dataset(test_data, window_size)

    jour_semaine_past_X, _ = create_dataset(data_past["jour_semaine"].values.reshape(-1, 1), window_size)
    jour_semaine_futur_X, _ = create_dataset(data_futur["jour_semaine"].values.reshape(-1, 1), window_size)

    jour_ferie_past_X, _ = create_dataset(data_past["jour_ferie"].values.reshape(-1, 1), window_size)
    jour_ferie_futur_X, _ = create_dataset(data_futur["jour_ferie"].values.reshape(-1, 1), window_size)

    train_X = np.stack((train_X, jour_semaine_past_X, jour_ferie_past_X), axis=2)
    test_X = np.stack((test_X, jour_semaine_futur_X, jour_ferie_futur_X), axis=2)

    # Mise en forme des données d'entrée pour LSTM (échantillons, pas de temps, fonctionnalités)
    # train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    # test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Créer un dossier pour sauvegarder le modèle
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    if is_model_load:
        model = load_model(os.path.join(model_dir, 'lstm_total_validation.h5'))
    else:
        # Construction du modèle LSTM
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(window_size, 3)))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='Adam')
        model.summary()


        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'lstm_total_validation.h5'),
                                                                 monitor='val_loss',
                                                                 save_best_only=True,
                                                                 mode='min')

        # Entraînement du modèle
        model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=100, batch_size=64, verbose=2, callbacks=[checkpoint_callback])

        model = load_model(os.path.join(model_dir, 'lstm_total_validation.h5'))

    # Prédiction sur l'ensemble de train
    predicted_train = model.predict(train_X)
    predicted_train = scaler.inverse_transform(predicted_train)
    predicted_train = predicted_train.flatten()

    # Prédiction sur l'ensemble de test
    predicted_test = model.predict(test_X)
    predicted_test = scaler.inverse_transform(predicted_test)
    predicted_test = predicted_test.flatten()
    # Comparaison des prédictions avec les données réelles
    train_values = scaler.inverse_transform(train_Y.reshape(-1, 1))
    train_values = train_values.flatten()
    test_values = scaler.inverse_transform(test_Y.reshape(-1, 1))
    test_values = test_values.flatten()

    incertitude_par_jour = np.zeros(7)

    corresponding_day = np.array(data_futur["jour_semaine"].tail(len(test_values)))
    for jour in range(7):
        test_values_tmp = test_values[corresponding_day == jour]
        predicted_test_tmp = predicted_test[corresponding_day == jour]
        incertitude_par_jour[jour] = np.mean(abs(test_values_tmp - predicted_test_tmp) / test_values_tmp)
    # incertitude = np.mean(abs(test_values.flatten() - predicted_test.flatten()) / test_values.flatten())
    print("Erreur moyenne en % sur le jeu de test")
    print(incertitude_par_jour)

    if plot:
        # Affichage des résultats train
        test_df = pd.DataFrame({'Actual': train_values, 'Predicted': predicted_train})
        test_df.plot()

        # Affichage des résultats test
        test_df = pd.DataFrame({'Actual': test_values, 'Predicted': predicted_test})
        test_df.plot()
        plt.show()

    # Prédiction pour les prochaines années
    # future_data = test_data[-window_size:]
    # future_data = np.reshape(future_data, (1, window_size, 1))
    # future_predictions = model.predict(future_data)
    # future_predictions = scaler.inverse_transform(future_predictions)
    # print("Prédictions pour les prochaines années :")
    # print(future_predictions)

    np.save(os.path.join(model_dir, "incertitudes_total.npy"), np.array([incertitude_par_jour]))

def model_titre_validation(data_path, is_model_load=True, plot=False):
    data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
    data = data[data.index <= "2019-12-01"]
    data['jour_semaine'] = data.index.strftime('%w').astype(int)
    france_holidays = holidays.France()
    data['jour_ferie'] = data.index.to_series().apply(lambda d: d in france_holidays).astype(int)
    data_past = data[data.index <= "2018-12-31"]
    data_futur = data[(data.index > "2018-12-31")]

    window_size = 45

    # Créer un dossier pour sauvegarder le scaler
    scaler_dir = 'model/scaler'
    os.makedirs(scaler_dir, exist_ok=True)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    unique_titre = np.load("model/CATEGORIE_TITRE.npy")
    titres_length = len(unique_titre)
    # unique_titre = np.char.replace(unique_titre, '?', 'INCONNU')
    for titre in unique_titre:
        data_past_filter = data_past[data_past["CATEGORIE_TITRE"] == titre]
        data_futur_filter = data_futur[data_futur["CATEGORIE_TITRE"] == titre]
        # Prétraitement des données
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data_past_filter['NB_VALD'].values.reshape(-1, 1))

        dump(scaler, os.path.join(scaler_dir, "scaler_titre_"+titre+".joblib"))

        train_data_tmp = scaler.transform(data_past_filter['NB_VALD'].values.reshape(-1, 1))
        test_data_tmp = scaler.transform(data_futur_filter['NB_VALD'].values.reshape(-1, 1))

        train_X_tmp, train_Y_tmp = create_dataset(train_data_tmp, window_size)
        test_X_tmp, test_Y_tmp = create_dataset(test_data_tmp, window_size)

        train_X.append(np.expand_dims(train_X_tmp, axis=2))
        train_Y.append(np.expand_dims(train_Y_tmp, axis=1))
        test_X.append(np.expand_dims(test_X_tmp, axis=2))
        test_Y.append(np.expand_dims(test_Y_tmp, axis=1))

    train_X = np.concatenate(train_X, axis=2)
    train_Y = np.concatenate(train_Y, axis=1)
    test_X = np.concatenate(test_X, axis=2)
    test_Y = np.concatenate(test_Y, axis=1)

    jour_semaine_past_X, _ = create_dataset(data_past_filter["jour_semaine"].values.reshape(-1, 1), window_size)
    jour_semaine_futur_X, _ = create_dataset(data_futur_filter["jour_semaine"].values.reshape(-1, 1), window_size)

    jour_ferie_past_X, _ = create_dataset(data_past_filter["jour_ferie"].values.reshape(-1, 1), window_size)
    jour_ferie_futur_X, _ = create_dataset(data_futur_filter["jour_ferie"].values.reshape(-1, 1), window_size)

    train_X = np.concatenate((train_X, jour_semaine_past_X[:, :, np.newaxis], jour_ferie_past_X[:, :, np.newaxis]), axis=2)
    test_X = np.concatenate((test_X, jour_semaine_futur_X[:, :, np.newaxis], jour_ferie_futur_X[:, :, np.newaxis]), axis=2)

    # Créer un dossier pour sauvegarder le modèle
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    if is_model_load:
        model = load_model(os.path.join(model_dir, 'lstm_titre_validation.h5'))
    else:
        # Construction du modèle LSTM
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(window_size, titres_length + 2)))
        # model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(LSTM(128))
        # model.add(Dropout(0.2))
        model.add(Dense(titres_length))
        model.compile(loss='mean_squared_error', optimizer='Adam')
        model.summary()

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'lstm_titre_validation.h5'),
                                                                 monitor='val_loss',
                                                                 save_best_only=True,
                                                                 mode='min')

        # Entraînement du modèle
        model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=100, batch_size=64, verbose=2, callbacks=[checkpoint_callback])

        model = load_model(os.path.join(model_dir, 'lstm_titre_validation.h5'))

    # Prédiction
    predicted_train = model.predict(train_X)
    predicted_test = model.predict(test_X)

    incertitude_par_titre_jour = np.zeros((titres_length, 7))

    for i, titre in enumerate(unique_titre):
        scaler = load(os.path.join(scaler_dir, "scaler_titre_"+titre+".joblib"))

        predicted_train_tmp = scaler.inverse_transform(predicted_train[:,i].reshape(-1, 1))
        predicted_train_tmp = predicted_train_tmp.flatten()
        predicted_test_tmp = scaler.inverse_transform(predicted_test[:,i].reshape(-1, 1))
        predicted_test_tmp = predicted_test_tmp.flatten()

        train_values = scaler.inverse_transform(train_Y[:,i].reshape(-1, 1))
        train_values = train_values.flatten()
        test_values = scaler.inverse_transform(test_Y[:,i].reshape(-1, 1))
        test_values = test_values.flatten()

        corresponding_day = np.array(data_futur_filter["jour_semaine"].tail(len(test_values)))
        for jour in range(7):
            test_values_tmp = test_values[corresponding_day == jour]
            predicted_test_tmp_tmp = predicted_test_tmp[corresponding_day == jour]
            incertitude_par_titre_jour[i, jour] = np.mean(abs(test_values_tmp - predicted_test_tmp_tmp) / test_values_tmp)

        if plot:
            # Affichage des résultats train
            test_df = pd.DataFrame({'Actual': train_values, 'Predicted': predicted_train_tmp})
            test_df.plot(title=titre)

            # Affichage des résultats test
            test_df = pd.DataFrame({'Actual': test_values, 'Predicted': predicted_test_tmp})
            test_df.plot(title=titre)
            plt.show()

        # Prédiction pour les prochaines années
        # future_data = test_data[-window_size:]
        # future_data = np.reshape(future_data, (1, window_size, 1))
        # future_predictions = model.predict(future_data)
        # future_predictions = scaler.inverse_transform(future_predictions)
        # print("Prédictions pour les prochaines années :")
        # print(future_predictions)

    print("Incertitude moyenne")
    print(np.mean(incertitude_par_titre_jour, axis=1))
    np.save(os.path.join(model_dir, "incertitudes_titre.npy"), np.array(incertitude_par_titre_jour))

def model_arret_validation(data_path, is_model_load=True, plot=False):
    data = pd.read_csv(data_path, sep=";", parse_dates=['JOUR'], index_col='JOUR')
    data = data[data.index <= "2019-12-01"]
    data['jour_semaine'] = data.index.strftime('%w').astype(int)
    france_holidays = holidays.France()
    data['jour_ferie'] = data.index.to_series().apply(lambda d: d in france_holidays).astype(int)
    data_past = data[data.index <= "2018-12-31"]
    data_futur = data[(data.index > "2018-12-31")]

    window_size = 45

    # Créer un dossier pour sauvegarder le scaler
    scaler_dir = 'model/scaler'
    os.makedirs(scaler_dir, exist_ok=True)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    unique_arret = np.load("model/LIBELLE_ARRET.npy")
    arrets_length = len(unique_arret)
    for arret in unique_arret:
        data_past_filter = data_past[data_past["LIBELLE_ARRET"] == arret]
        data_futur_filter = data_futur[data_futur["LIBELLE_ARRET"] == arret]
        # Prétraitement des données
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data_past_filter['NB_VALD'].values.reshape(-1, 1))

        dump(scaler, os.path.join(scaler_dir, "scaler_titre_"+arret+".joblib"))

        train_data_tmp = scaler.transform(data_past_filter['NB_VALD'].values.reshape(-1, 1))
        test_data_tmp = scaler.transform(data_futur_filter['NB_VALD'].values.reshape(-1, 1))

        train_X_tmp, train_Y_tmp = create_dataset(train_data_tmp, window_size)
        test_X_tmp, test_Y_tmp = create_dataset(test_data_tmp, window_size)

        train_X.append(np.expand_dims(train_X_tmp, axis=2))
        train_Y.append(np.expand_dims(train_Y_tmp, axis=1))
        test_X.append(np.expand_dims(test_X_tmp, axis=2))
        test_Y.append(np.expand_dims(test_Y_tmp, axis=1))

    train_X = np.concatenate(train_X, axis=2)
    train_Y = np.concatenate(train_Y, axis=1)
    test_X = np.concatenate(test_X, axis=2)
    test_Y = np.concatenate(test_Y, axis=1)

    jour_semaine_past_X, _ = create_dataset(data_past_filter["jour_semaine"].values.reshape(-1, 1), window_size)
    jour_semaine_futur_X, _ = create_dataset(data_futur_filter["jour_semaine"].values.reshape(-1, 1), window_size)

    jour_ferie_past_X, _ = create_dataset(data_past_filter["jour_ferie"].values.reshape(-1, 1), window_size)
    jour_ferie_futur_X, _ = create_dataset(data_futur_filter["jour_ferie"].values.reshape(-1, 1), window_size)

    train_X = np.concatenate((train_X, jour_semaine_past_X[:, :, np.newaxis], jour_ferie_past_X[:, :, np.newaxis]),axis=2)
    test_X = np.concatenate((test_X, jour_semaine_futur_X[:, :, np.newaxis], jour_ferie_futur_X[:, :, np.newaxis]),axis=2)

    # Créer un dossier pour sauvegarder le modèle
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    if is_model_load:
        model = load_model(os.path.join(model_dir, 'lstm_arret_validation.h5'))
    else:
        # Construction du modèle LSTM
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(window_size, arrets_length + 2)))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(arrets_length))
        model.compile(loss='mean_squared_error', optimizer='Adam')
        model.summary()

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'lstm_arret_validation.h5'),
                                                                 monitor='val_loss',
                                                                 save_best_only=True,
                                                                 mode='min')

        # Entraînement du modèle
        model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=100, batch_size=64, verbose=2, callbacks=[checkpoint_callback])

        model = load_model(os.path.join(model_dir, 'lstm_arret_validation.h5'))

    # Prédiction
    predicted_train = model.predict(train_X)
    predicted_test = model.predict(test_X)

    incertitude_par_arret_jour = np.zeros((arrets_length, 7))

    for i, arret in enumerate(unique_arret):
        scaler = load(os.path.join(scaler_dir, "scaler_titre_"+arret+".joblib"))

        predicted_train_tmp = scaler.inverse_transform(predicted_train[:,i].reshape(-1, 1))
        predicted_train_tmp = predicted_train_tmp.flatten()
        predicted_test_tmp = scaler.inverse_transform(predicted_test[:,i].reshape(-1, 1))
        predicted_test_tmp = predicted_test_tmp.flatten()

        train_values = scaler.inverse_transform(train_Y[:,i].reshape(-1, 1))
        train_values = train_values.flatten()
        test_values = scaler.inverse_transform(test_Y[:,i].reshape(-1, 1))
        test_values = test_values.flatten()

        corresponding_day = np.array(data_futur_filter["jour_semaine"].tail(len(test_values)))
        for jour in range(7):
            test_values_tmp = test_values[corresponding_day == jour]
            predicted_test_tmp_tmp = predicted_test_tmp[corresponding_day == jour]
            incertitude_par_arret_jour[i, jour] = np.mean(abs(test_values_tmp - predicted_test_tmp_tmp) / test_values_tmp)


        if plot:
            # Affichage des résultats train
            test_df = pd.DataFrame({'Actual': train_values, 'Predicted': predicted_train_tmp})
            test_df.plot(title=arret)

            # Affichage des résultats test
            test_df = pd.DataFrame({'Actual': test_values, 'Predicted': predicted_test_tmp})
            test_df.plot(title=arret)
            plt.show()

        # Prédiction pour les prochaines années
        # future_data = test_data[-window_size:]
        # future_data = np.reshape(future_data, (1, window_size, 1))
        # future_predictions = model.predict(future_data)
        # future_predictions = scaler.inverse_transform(future_predictions)
        # print("Prédictions pour les prochaines années :")
        # print(future_predictions)

    print("Incertitude moyenne")
    print(np.mean(incertitude_par_arret_jour, axis=1))
    np.save(os.path.join(model_dir, "incertitudes_arret.npy"), np.array(incertitude_par_arret_jour))


# find_and_save_complete_name("data/past/validation_groupby_JOUR_CATEGORIE_TITRE.csv", "CATEGORIE_TITRE")
# find_and_save_complete_name("data/past/validation_groupby_JOUR_LIBELLE_ARRET.csv", "LIBELLE_ARRET")
# model_total_validation("data/past/validation_groupby_JOUR.csv", is_model_load=False, plot=True)
# model_titre_validation("data/past/validation_groupby_JOUR_CATEGORIE_TITRE.csv", is_model_load=False, plot=False)
# model_arret_validation("data/past/validation_groupby_JOUR_LIBELLE_ARRET.csv", is_model_load=False, plot=False)

# print(len(np.load("model/LIBELLE_ARRET.npy")))
