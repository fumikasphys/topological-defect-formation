import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Activation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import math

def scaled_softsign(x):
    return tf.math.sqrt(2.0) * tf.math.divide(x, 1 + tf.math.abs(x))


vector_length =1024


# Name of the folder where the data are saved
folder_path = 'j'


data_list = []


for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  
        file_path = os.path.join(folder_path, filename)
        

        
        with open(file_path, 'r') as file:
            data_string = file.read().strip()  
            
            
            data = ast.literal_eval(data_string)
            
            
            data_list.append(data)

data2 = np.array(data_list)


X = data2[:, :-1, :] # the short time-series input
Y = data2[:, -1, :]  # final defect configuration

# Split into training and test sets (e.g., 80% train, 20% test)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Replace SimpleRNN with LSTM or Transformer to compare

model = Sequential()
model.add(SimpleRNN(256,  input_shape=(X.shape[1], X.shape[2]), return_sequences=False,activation='softsign'))
model.add(Dense(X.shape[2]))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=60, batch_size=10, verbose=50)





plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




