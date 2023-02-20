import tensorflow as tf

PARAMS = {}
PARAMS['PATCH_SIZE'] = 25
PARAMS['NUM_CHANNELS'] = 3
PARAMS['IMAGE_SIZE'] = 200
PARAMS['NUM_PATCHES'] = PARAMS['IMAGE_SIZE'] ** 2 // PARAMS['PATCH_SIZE'] ** 2


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, NUM_PATCHES, HIDDEN_SIZE):
        super().__init__(name='custom_conv2d')
        self.linear_projection = tf.keras.layers.Dense(HIDDEN_SIZE)
        self.positional_embedding = tf.keras.layers.Embedding(NUM_PATCHES, HIDDEN_SIZE)
        self.NUM_PATCHES = NUM_PATCHES

    def call(self, x):
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], 1],
            strides=[1, PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, (tf.shape(patches)[0], -1, patches.shape[-1]))
        embedding_input = tf.range(start=0, limit=self.NUM_PATCHES, delta=1)
        output = self.linear_projection(patches) + self.positional_embedding(embedding_input)
        return output


patch_encoder = PatchEncoder(64, 1875)
print(patch_encoder(tf.zeros([1, 200, 200, 3])))
