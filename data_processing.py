import pandas as pd
import numpy as np


csv_file = pd.read_csv("./data/data.csv").values

# Shuffling the data to not overfeed our model
np.random.shuffle(csv_file)
#Prepare the data from the labels
x_data = np.concatenate((csv_file[:, 0].reshape(-1, 1), csv_file[:, 2:9], csv_file[:, 10].reshape(-1, 1)),axis=1)
y_data = csv_file[:, 9]

#Separate the training data from the test data
x_train = x_data[:int(0.8 * len(x_data)), :]
y_train = y_data[:int(0.8 * len(y_data))]

x_test = x_data[int(0.8 * len(x_data)):, :]
y_test = y_data[int(0.8 * len(y_data)):]