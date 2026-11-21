# Student Name: Seth McKenzie
# UCID: 30242422
# CPSC 433 F26 Tut02
# Original Author: Jonathan Hudson

import tensorflow as tf
import numpy as np

tf.random.set_seed(1234) # fixed seed for reproducibility

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
# Scale pixel values from 0-255 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel axis so images are (28, 28, 1), as Conv2D layers expect
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#Augemnation: add an inverted copy of every training image so the model sees white on black and black on white
x_aug = np.concatenate([x_train, 1.0 - x_train])
y_aug = np.concatenate([y_train, y_train])

#Shuffle so validation_split doesn't take only the inverted copies
rng = np.random.default_rng(1234)
perm = rng.permutation(len(x_aug))
x_aug, y_aug = x_aug[perm], y_aug[perm]
print(x_aug.shape)

print("--Make model--")
# CNN: Filters detect local strokes and curves and share weights across the image
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same'), # 32 3*3 filters
  tf.keras.layers.MaxPooling2D(), # Halves the image size
  tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'), # 64 3*3 filters
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dropout(0.3), # Randomly turn off 30% of input units to reduce overfitting
  tf.keras.layers.Dense(10, activation = 'softmax') # one probability per letter A-J
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("--Fit model--")
# Stop when vlaidation loss hasn't improved for 3 epochs and restore the previous best weights
early = tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)
model.fit(x_aug, y_aug, epochs=20, batch_size=128, validation_split=0.1, callbacks=[early], verbose=2)

print("--Evaluate model--")
loss_tr, acc_tr = model.evaluate(x_train,  y_train, verbose=2)
loss_te, acc_te = model.evaluate(x_test,  y_test, verbose=2)
loss_inv, acc_inv = model.evaluate(1.0 - x_test, y_test, verbose=2) # Test images with polarity inverted
print(f"Train / Test / Inverted-test: {acc_tr*100:.1f}% / {acc_te*100:.1f}% / {acc_inv*100:.1f}%")

model.save("notMNIST-Complete.keras") # Save model