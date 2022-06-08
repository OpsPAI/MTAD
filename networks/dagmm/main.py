import tensorflow as tf
from DAGMM.dagmm import DAGMM
import numpy as np
import pandas as pd
import os

# Initialize
model = DAGMM(
    comp_hiddens=[32, 16, 2], comp_activation=tf.nn.tanh,
    est_hiddens=[80, 40], est_activation=tf.nn.tanh,
    est_dropout_ratio=0.25
)

# Fit the training data to model
data_dir_path = 'C:/Users/Administrator/Downloads/DAGMM-master/SMD/data_concat/'
csvs = os.listdir(data_dir_path)

csv_path = []

for i in csvs:
    csv_path.append(data_dir_path + i)

numbers = []

for j in csvs:
    name_temp = os.path.split(j)[1]
    numbers.append(name_temp[5:-4])


def generate_score(number):
    # Read the raw data.
    input_dir_path = 'C:/Users/Administrator/Downloads/DAGMM-master/SMD/data_concat/data-' + number + '.csv'
    data = np.array(pd.read_csv(input_dir_path, header=None), dtype=np.float64)
    x_train = data[: len(data) // 2]
    x_test = data[len(data) // 2:]
    print(len(x_train))
    print(len(x_test))
    model.fit(x_train)
    if not os.path.exists('../score'):
        os.makedirs('../score')
    # Evaluate energies
    # (the more the energy is, the more it is anomaly)
    energy = model.predict(x_test)
    np.save('../score/' + number + '.npy', energy)
    # Save fitted model to the directory
    model.save('./model/fitted_model' + number)

    # Restore saved model from directory
    model.restore('./model/fitted_model' + number)


for j in numbers:
    generate_score(j)
    print('Finish generating ' + j)
