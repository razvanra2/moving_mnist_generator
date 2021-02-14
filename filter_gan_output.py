import tensorflow as tf
import os
from PIL import Image
from numpy import asarray
import numpy as np
from shutil import copy2
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

gan_imgs_array = []
gan_filenames_array = []
for filename in os.listdir('./gans/results/'):
    img = Image.open(f'./gans/results/{filename}').convert("L")
    img_data = (asarray(img) / 255.0).flatten()
    gan_imgs_array.append(img_data)
    gan_filenames_array.append(f'./gans/results/{filename}')

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6)

print(model.metrics_names)
print(model.evaluate(x_test,  y_test, verbose=1000))


predi = model.predict(np.array([x_test[0]]))
print(predi)
file_cnt = 0
for img in gan_imgs_array:
    res = model.predict(np.array([img]))[0]
    for val in res:
        if (val > 0.7):
            print(gan_filenames_array[file_cnt])
            copy2(gan_filenames_array[file_cnt], './gans/filtered_results/')
        break
    file_cnt += 1
