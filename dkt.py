import re

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential

"""
Criação da lista para substituição nas equações
"""
lista = []
c = 0
for i in range(ord('A'), ord("Z")+1):
    lista.insert(c, chr(i))
    c += 1
    for j in range(ord('A'), ord('Z')+1):
        lista.append(chr(i)+chr(j))

"""
Função para a simplificação das equações e passos
"""
def simplifica_equação(equ):
    i = 0
    for m in re.finditer(r"(([\d]{0,}x{1})|([\d]{1,}))", equ):
        equ = equ.replace(m.group(0).replace("x", ""), lista[i], 1)
        i = i + 1
    return equ

"""
Função para criar do One Hot Encode
"""
def get_OHE(df):
    df_OHE = pd.concat([df[['e-mail', 'AD', 'SB', "MT", "DV", "SP", "PA", "PM", "MM", "DM", "AF", "MF", "DF", "OI", "UT", "RE", "ER", "DE"]], 
                        pd.get_dummies(df[['exercício', 'passo']], drop_first=False)], axis=1)
    return df_OHE


df = pd.read_csv("data/Dados_DKT.csv", sep=";", low_memory=False)


"""
Pré-processamento
"""
df = df.drop('feedback', axis=1)
df = df.drop('message', axis=1)

df['exercício'] = df['exercício'].map(simplifica_equação)
df['passo'] = df['passo'].map(simplifica_equação)

correto = df.iloc[:, 20].values
df = df.drop('correct', axis=1)

"""
processor = make_column_transformer(
    (OneHotEncoder(), ['exercício', 'passo']),
)

previsores = processor.fit_transform(df).toarray()
"""
label_encoder = LabelEncoder()

df['exercício'] = label_encoder.fit_transform(df['exercício'])
df['passo'] = label_encoder.fit_transform(df['passo'])

previsores = df.values
#previsores = get_OHE(df)

previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))


"""
Rede Neural
"""
regressor = Sequential()
regressor.add(LSTM(units=20, return_sequences=True, input_shape=(previsores.shape[1],1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=20, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=20, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=8))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear'))

regressor.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy', AUC()])

regressor.fit(previsores, correto, batch_size=512, epochs=5)
