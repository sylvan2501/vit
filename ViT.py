import tensorflow as tf

from PatchEncoder import PatchEncoder
from TransformerEncoder import TransformerEncoder

PARAMS = {}
PARAMS['NUM_CLASSES'] = 2


class ViT(tf.keras.Model):
    def __init__(self, NUM_HEADS, HIDDEN_SIZE, NUM_PATCHES, NUM_LAYERS, NUM_DENSE_UNITS, NUM_CLASSES):
        super().__init__(name='vision_transformer')
        self.patch_encoder = PatchEncoder(NUM_PATCHES, HIDDEN_SIZE)
        self.trans_encoder = [TransformerEncoder(NUM_HEADS, HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        self.dense_one = tf.keras.layers.Dense(NUM_DENSE_UNITS, tf.nn.gelu)
        self.dense_two = tf.keras.layers.Dense(NUM_DENSE_UNITS, tf.nn.gelu)
        self.dense_three = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        self.NUM_LAYERS = NUM_LAYERS

    def call(self, input, training=True):
        x = self.patch_encoder(input)
        for i in range(self.NUM_LAYERS):
            x = self.trans_encoder[i](x)

        x = tf.keras.layers.Flatten()(x)
        x = self.dense_one(x)
        x = self.dense_two(x)

        return self.dense_three(x)


vit = ViT(NUM_HEADS=4, HIDDEN_SIZE=1875, NUM_PATCHES=64, NUM_LAYERS=2, NUM_DENSE_UNITS=128, NUM_CLASSES=1)
print(vit(tf.zeros([2, 200, 200, 3])))
