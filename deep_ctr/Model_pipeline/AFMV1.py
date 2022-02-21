from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.initializers import glorot_normal, Zeros
import tensorflow as tf
import itertools


class afm(Layer):
    def __init__(self, attention_size=4, seed=1024, **kwargs):
        self.seed = seed
        self.attention_size = attention_size
        super(afm, self).__init__(**kwargs)
        
    def build(self, input_shape):
        embed_size = input_shape[-1].value
        self.att_w = self.add_weight(name='att weights', shape=(embed_size, self.attention_size), initializer=glorot_normal(self.seed))
        self.att_b = self.add_weight(name='att bias', shape=(self.attention_size, ), initializer=Zeros())
        self.projection_h = self.add_weight(name='projection_h', shape=(self.attention_size, 1), initializer=glorot_normal(self.seed))
        self.projection_p = self.add_weight(name='projection_p', shape=(embed_size, 1), initializer=Zeros())
        self.tensordot = tf.keras.layers.Lambda(lambda x : tf.tensordot(x[0], x[1], axes=(-1, 0)))
        super(afm, self).build(input_shape)
        
    def call(self, inputs):
        embed_vec_list = inputs
        row = []
        col = []
        for r, w in itertools.combinations(embed_vec_list, 2):
            row.append(r)
            col.append(w)
        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)
        inner_product = p * q
        att_tmp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(inner_product, self.att_w, axes=(-1, 0)), self.att_b))
        self.att_normalized = tf.nn.softmax(tf.tensordot(att_tmp, self.projection_h, axes=[-1, 0]), dim=1)
        att_output = tf.reduce_sum(self.att_normalized * inner_product, axis=1)
        att_output = tf.keras.layers.Dropout(0.2, seed=self.seed)
        afm_out = tf.tensordot(att_output, self.projection_p)
        return afm_out