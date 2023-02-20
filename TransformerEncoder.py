import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, NUM_HEADS, HIDDEN_SIZE):
        super().__init__(name='transformer_encoder')

        self.first_norm = tf.keras.layers.LayerNormalization()
        self.second_norm = tf.keras.layers.LayerNormalization()

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(NUM_HEADS, HIDDEN_SIZE)
        self.dense_one = tf.keras.layers.Dense(HIDDEN_SIZE, activation=tf.nn.gelu)
        self.dense_two = tf.keras.layers.Dense(HIDDEN_SIZE, activation=tf.nn.gelu)

    def call(self, input):
        x_one = self.first_norm(input)
        x_one = self.multi_head_attention(x_one, x_one)
        x_one = tf.keras.layers.Add()([x_one, input])
        x_two = self.second_norm(x_one)
        x_two = self.dense_one(x_two)
        output = self.dense_two(x_two)
        output = tf.keras.layers.Add()([output, x_one])

        return output


transformer_encoder = TransformerEncoder(8, 1875)
print(transformer_encoder(tf.zeros([1, 625, 1875])))
