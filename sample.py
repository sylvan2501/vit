import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
PARAMS = {}
PARAMS['PATCH_SIZE'] = 25
PARAMS['NUM_CHANNELS'] = 3
PARAMS['IMAGE_SIZE'] = 200
PARAMS['NUM_PATCHES'] = PARAMS['IMAGE_SIZE']**2 // PARAMS['PATCH_SIZE']**2
image_path = './5673551_01d1ea993e_n.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (PARAMS['IMAGE_SIZE'], PARAMS['IMAGE_SIZE']))
patches = tf.image.extract_patches(
    images=tf.expand_dims(image, axis=0),
    sizes=[1, PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], 1],
    strides=[1, PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)

print(patches.shape)
patches = tf.reshape(patches, (patches.shape[0], -1, 1875))
print(patches.shape)
plt.figure(figsize=(8, 8))


k = 0

for i in range(PARAMS['IMAGE_SIZE']**2//PARAMS['PATCH_SIZE']**2):
    ax = plt.subplot(PARAMS['IMAGE_SIZE']//PARAMS['PATCH_SIZE'], PARAMS['IMAGE_SIZE']//PARAMS['PATCH_SIZE'], i+1)
    plt.imshow(tf.reshape(patches[0, i, :], shape=(PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], PARAMS['NUM_CHANNELS'])))
    plt.axis('off')
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(PARAMS['NUM_PATCHES'], PARAMS['PATCH_SIZE']*PARAMS['PATCH_SIZE']*PARAMS['NUM_CHANNELS']))
input_array = np.random.randint(PARAMS['NUM_PATCHES'], size=(1, PARAMS['NUM_PATCHES']))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape)
