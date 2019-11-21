from __future__ import division
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import initializers, activations


class AttentiveConvLSTM(Layer):
    def __init__(self, nb_filters_in, nb_filters_out, nb_filters_at,
                 nb_rows, nb_cols, init='normal', inner_init='orthogonal',
                 attentive_init='zero', activation='tanh', inner_activation='sigmoid',
                 W_regularizer=None, U_regularizer=None, weights=None, go_backwards=False,
                 ):
        super().__init__()
        self.nb_filters_in  = nb_filters_in
        self.nb_filters_out = nb_filters_out
        self.nb_filters_at = nb_filters_at

        self.nb_rows = nb_rows
        self.nb_cols  = nb_cols

        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)
        self.attentive_init = initializers.get(attentive_init)

        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.initial_weights = weights
        self.go_backwards = go_backwards

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.input_spec = [InputSpec(ndim=5)]

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.nb_filters_out,) + input_shape[3:]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_initial_states(self, x):
        initial_state = K.sum(x, axis=1)
        initial_state = K.conv2d(initial_state,
                                 K.zeros((self.nb_filters_out, self.nb_filters_in, 1, 1)),
                                 padding='same')
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.states = [None, None]

        self.W_a = Conv2D(self.nb_filters_at, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.U_a = Conv2D(self.nb_filters_at, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.V_a = Conv2D(1, self.nb_rows, self.nb_cols, padding='same',
                          use_bias=True)

        self.W_a.build((input_shape[0], self.nb_filters_at,
                        input_shape[3], input_shape[4]))
        self.U_a.build((input_shape[0], self.nb_filters_in,
                        input_shape[3], input_shape[4]))
        self.V_a.build((input_shape[0], self.nb_filters_at,
                        input_shape[3], input_shape[4]))

        self.W_a.built = True
        self.U_a.built = True
        self.V_a.build = True

        self.W_i = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.U_i = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)

        self.W_i.build((input_shape[0], self.nb_filters_in,
                        input_shape[3], input_shape[4]))
        self.U_i.build((input_shape[0], self.nb_filters_out,
                        input_shape[3], input_shape[4]))

        self.W_i.built = True
        self.U_i.built = True

        self.W_f = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.U_f = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)

        self.W_f.build((input_shape[0], self.nb_filters_in,
                        input_shape[3], input_shape[4]))
        self.U_f.build((input_shape[0], self.nb_filters_out,
                        input_shape[3], input_shape[4]))

        self.W_f.built = True
        self.U_f.built = True

        self.W_c = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.U_c = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)

        self.W_c.build((input_shape[0], self.nb_filters_in,
                        input_shape[3], input_shape[4]))
        self.U_c.build((input_shape[0], self.nb_filters_out,
                        input_shape[3], input_shape[4]))

        self.W_c.built = True
        self.U_c.built = True

        self.W_o = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.U_o = Conv2D(self.nb_filters_out, self.nb_rows, self.nb_cols,
                          padding='same', use_bias=True)
        self.W_o.built = True
        self.U_o.built = True

        self.trainable_weights = []
        self.trainable_weights.extend(self.W_a.trainable_weights)
        self.trainable_weights.extend(self.U_a.trainable_weights)
        self.trainable_weights.extend(self.V_a.trainable_weights)

        self.trainable_weights.extend(self.W_i.trainable_weights)
        self.trainable_weights.extend(self.U_i.trainable_weights)

        self.trainable_weights.extend(self.W_f.trainable_weights)
        self.trainable_weights.extend(self.U_f.trainable_weights)

        self.trainable_weights.extend(self.W_c.trainable_weights)
        self.trainable_weights.extend(self.U_c.trainable_weights)

        self.trainable_weights.extend(self.W_o.trainable_weights)
        self.trainable_weights.extend(self.U_o.trainable_weights)

    def get_constants(self, x):
        return []

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        x_shape = K.shape(x)
        h_tm1 = states[0]
        c_tm1 = states[1]

        e = self.V_a(K.tanh(self.W_a(h_tm1) + self.U_a(x)))
        a = K.reshape(K.softmax(K.batch_flatten(e)),
                      (x_shape[0], 1, x_shape[2], x_shape[3]))
        x_tilde = x * K.repeat_elements(a, x_shape[1], 1)

        x_i = self.W_i(x_tilde)
        x_f = self.W_f(x_tilde)
        x_c = self.W_c(x_tilde)
        x_o = self.W_o(x_tilde)

        i = self.inner_activation(x_i + self.U_i(h_tm1))
        f = self.inner_activation(x_f + self.U_f(h_tm1))
        c = f * c_tm1 + i * self.activation(x_c + self.U_c(h_tm1))
        o = self.inner_activation(x_o + self.U_o(h_tm1))

        h = o * self.activation(c)

        return h, [h, c]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=False,
                                             mask=mask,
                                             constants=constants,
                                             unroll=False,
                                             input_length=input_shape[1])

        if last_output.ndim == 3:
            last_output = K.expand_dims(last_output, dim=0)

        return last_output