import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential([
    Dense(200, activation='relu', input_shape=(32, 32, 3)),
    Flatten(),
    Dense(150, activation='relu'),
    Dense(150, activation='softmax')
])

print(model.summary())