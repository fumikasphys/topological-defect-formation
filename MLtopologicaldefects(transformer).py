import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MultiHeadAttention, LayerNormalization, Embedding
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import math



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



# Transformer version


hidden_dim = 256
num_heads = 4


class PositionalEmbedding(Layer):

    def __init__(self, sequence_length, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=hidden_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.position_embedding(positions)


class TakeLastTimeStep(Layer):

    def call(self, x):
        return x[:, -1, :]


inputs = tf.keras.Input(
    shape=(X.shape[1], X.shape[2]))


x = Dense(
    hidden_dim,
    activation='softsign')(inputs)


x = PositionalEmbedding(
    sequence_length=X.shape[1],
    hidden_dim=hidden_dim)(x)


attention_output = MultiHeadAttention(
    num_heads=num_heads,
    key_dim=hidden_dim // num_heads)(x, x)


x = LayerNormalization(epsilon=1e-6)(x + attention_output)


ff = Dense(
    hidden_dim,
    activation='softsign')(x)


x = LayerNormalization(epsilon=1e-6)(x + ff)


x = TakeLastTimeStep()(x)


outputs = Dense(X.shape[2])(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)


model.compile(optimizer='adam',loss='mean_squared_error')


history = model.fit(X_train,Y_train,validation_data=(X_val, Y_val),epochs=60,batch_size=10,verbose=2)




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
