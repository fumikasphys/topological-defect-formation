# Topological Defect Prediction with Recurrent Neural Networks

This repository contains Python scripts for generating topological-defect data from a one-dimensional stochastic Ginzburg–Landau model and training a neural network to predict the final defect configuration from a short time series.

## Files

- `topologicaldefects.py`  
  Generates independent realizations of the field dynamics and saves a short time series together with the final field configuration.

- `MLtopologicaldefects.py`  
  Loads the generated data, separates the short time-series input from the final configuration, trains a Simple RNN, and plots the training and validation loss.

## Data generation

The simulation uses a one-dimensional lattice with

```python
L = 512
n0 = 1024
dx = L / n0
tauQ = 128
```

The number of independent samples is controlled by

```python
Nsamples = 10
```

For example, set

```python
Nsamples = 3000
```

to generate 3000 samples.

For each realization, the code saves 11 field configurations from the short time interval

```text
2370, 2380, 2390, ..., 2470
```

followed by the final field configuration at

```text
3200
```

Thus, each sample contains 12 vectors in total:

- 11 vectors used as the time-series input;
- 1 vector used as the final prediction target.

Each vector contains 1024 spatial lattice values.

## Data format

The machine-learning script constructs

```python
X = data2[:, :-1, :]   # short time-series input
Y = data2[:, -1, :]    # final field configuration
```

Therefore, for `Nsamples = 3000`,

```text
X.shape = (3000, 11, 1024)
Y.shape = (3000, 1024)
```

## Folder structure

The machine-learning script expects the data files to be stored in a folder named

```text
j/
```

For example:

```text
project/
├── README.md
├── topologicaldefects(samples)(1).py
├── MLtopologicaldefects.py
└── j/
    ├── j1.txt
    ├── j2.txt
    ├── j3.txt
    └── ...
```

The current data-generation script writes files as `j1.txt`, `j2.txt`, etc. in the working directory. Before running the machine-learning script, either move these files into the `j/` folder or modify the output line to save them directly there.

For example:

```python
import os

os.makedirs("j", exist_ok=True)

with open(f"j/j{j}.txt", "w") as file:
    ...
```

## Neural-network model

The baseline model is a Simple RNN:

```python
model = Sequential()
model.add(
    SimpleRNN(
        256,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=False,
        activation="softsign"
    )
)
model.add(Dense(X.shape[2]))
model.compile(optimizer="adam", loss="mean_squared_error")
```

The data are split into training and validation sets using an 80/20 split.

The model is trained for 60 epochs with batch size 10:

```python
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=60,
    batch_size=10
)
```

The Simple RNN can be replaced with an LSTM or a Transformer architecture to compare.

## Requirements

The scripts require Python 3 and the following packages:

```text
numpy
matplotlib
tensorflow
scikit-learn
```

Install them with

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

## Running the code

### 1. Generate data

Run

```bash
python "topologicaldefects(samples)(1).py"
```

Make sure the generated `j*.txt` files are placed in the `j/` folder.

### 2. Train the neural network

Run

```bash
python MLtopologicaldefects.py
```

The script trains the network and plots the training and validation mean-squared-error loss as a function of epoch.


