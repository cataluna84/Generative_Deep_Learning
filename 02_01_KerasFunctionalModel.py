from threading import active_count

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.datasets import cifar10

NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Converting to one-hot encoding
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

input_layer = Input(shape=(32, 32, 3))

x = Flatten()(input_layer)

x = Dense(units=200, activation='relu')(x)
x = Dense(units=150, activation='relu')(x)

output_layer = Dense(units=NUM_CLASSES, activation='softmax')(x)

model = Model(input_layer, output_layer)

# print(model.summary())
opt = Adam(learning_rate=0.00005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x = x_train, y = y_train, batch_size = 32, epochs = 20, shuffle = True)

model.evaluate(x = x_test, y = y_test)

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x = x_test)
print(preds.shape)
preds_single = CLASSES[np.argmax(a = preds, axis= -1)]
actual_single = CLASSES[np.argmax(a = y_test, axis = -1)]

n_to_show = 10
indices = np.random.choice(a = range(len(x_test)), size = n_to_show)

fig = plt.figure(figsize=(20, 6))
fig.subplots_adjust(hspace = 0.6, wspace = 0.5)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)

plt.show()