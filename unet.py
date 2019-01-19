# -*- coding: utf-8 -*-
from loader import load_data
import numpy as np
# Data loading
(X_train, y_train, X_test, y_test) = load_data()

# Data showing
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(X_train[300])
plt.subplot(1,2,2)
plt.imshow(np.argmax(y_train[300], axis=2))

plt.show()
plt.clf()

plt.subplot(1,2,1);
plt.imshow(X_train[20]);
plt.subplot(1,2,2);
plt.imshow(np.argmax(y_train[20], axis=2))

plt.show()
plt.clf()

# Carga del modelo
from model.u_net import get_unet_128 as unet

model = unet(num_classes=20)
print(model.summary())

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
            
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               min_delta=1e-4)]

entrenamiento = model.fit(X_train, y_train, batch_size=16, epochs=100, callbacks=callbacks, validation_data=(X_test, y_test))          

model.save_weights('model_weights.h5')

ent_loss = entrenamiento.history['loss']
val_loss = entrenamiento.history['val_loss']

epochs = range(1, len(ent_loss) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, ent_loss, 'b', label='Entrenamiento')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'r', label='Validación')
plt.title('Pérdida en Entrenamiento y Validación')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

predicted = model.predict(X_test)

fig = plt.figure(figsize=(8,8))
fig.add_subplot(1, 3,1)
plt.imshow(X_test[0])
fig.add_subplot(1,3 ,2)
plt.imshow(np.argmax(y_test[0], axis=2))
fig.add_subplot(1,3,3)
plt.imshow(np.argmax(predicted[0], axis=2))

fig.show()
