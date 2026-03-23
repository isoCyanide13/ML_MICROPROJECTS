import os
import pickle
import numpy as np
import tensorflow as tf
from keras import Model, ops
from keras import random as rn
from keras.layers import (Input, Conv2D, ReLU, BatchNormalization,
                          Flatten, Dense, Reshape, Conv2DTranspose,
                          Activation, Layer)
from keras.optimizers import Adam

"""
    Sampling Layer — Reparameterization Trick
    -----------------------------------------
    In a VAE, we can't backpropagate through a random sampling operation directly
    because randomness has no gradient. The reparameterization trick solves this
    by separating the randomness (epsilon) from the learnable parameters (mu, log_variance).

    Instead of sampling z directly from N(mu, sigma), we sample epsilon from N(0,1)
    and compute z = mu + sigma * epsilon. This way the randomness is in epsilon
    (a fixed external noise source) and the gradients can flow through mu and sigma
    during backpropagation.
"""
class Sampling(Layer):
    def call(self, inputs):
        mu, log_variance = inputs
        # This is the "noise" that makes each forward pass slightly different
        # giving the VAE its generative capability
        epsilon = rn.normal(shape=ops.shape(mu), mean=0.0, stddev=1.0)
        return mu + ops.exp(log_variance * 0.5) * epsilon

"""
    VAEModel — Custom Keras Model with Manual Training Loop
    --------------------------------------------------------
    We subclass keras.Model instead of using the standard Functional API because
    we need a custom train_step. The standard model.compile(loss=...) approach
    expects loss functions with (y_true, y_pred) signatures, but KL loss doesn't
    depend on y_true or y_pred at all — it only depends on mu and log_variance
    from the encoder. A custom train_step gives us full control over how the
    loss is computed and how gradients are applied each batch.
"""
class VAEModel(Model):
    def __init__(self,
                 encoder,
                 decoder, 
                 reconstruction_loss_weight,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def train_step(self, data):
        """
        Called automatically by Keras once per batch during model.fit().
        Replaces the default Keras training loop with full manual control.
        Everything inside GradientTape is recorded for backpropagation.
        """
        x = data[0] if isinstance(data, tuple) else data
        
        """
        GradientTape — Automatic Differentiation
        -----------------------------------------
        GradientTape records every mathematical operation performed inside
        its context block. When we call tape.gradient(loss, weights) later,
        it replays these operations in reverse (chain rule) to compute
        how much each weight contributed to the loss. This is backpropagation.
        """
        with tf.GradientTape() as tape:
            mu, log_variance, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)
            
            
            reconstruction_loss = ops.mean(
                ops.sum(ops.square(x - reconstruction), axis=[1, 2, 3])
            )
            kl_loss = -0.5 * ops.mean(
                ops.sum(1 + log_variance
                        - ops.square(mu)
                        - ops.exp(log_variance), axis=1)
            )
            total_loss = (self.reconstruction_loss_weight * reconstruction_loss) + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }


class VAE:
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 100

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def compile(self, learning_rate=0.0001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate))

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def save(self, save_folder=r"D:\ML_MICROPROJECTS\My_Trained_Models"):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        # weights_path here is the folder, not a single file
        encoder_path = os.path.join(weights_path, "encoder.weights.h5")
        decoder_path = os.path.join(weights_path, "decoder.weights.h5")
        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)

    def reconstruct(self, images):
        mu, log_variance, z = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(mu)
        return reconstructed_images, mu, log_variance

    @classmethod
    def load(cls, save_folder=r"D:\ML_MICROPROJECTS\My_Trained_Models"):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        autoencoder.load_weights(save_folder)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        encoder_path = os.path.join(save_folder, "encoder.weights.h5")
        decoder_path = os.path.join(save_folder, "decoder.weights.h5")
        self.encoder.save_weights(encoder_path)
        self.decoder.save_weights(decoder_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        self.model = VAEModel(
            encoder=self.encoder,
            decoder=self.decoder,
            reconstruction_loss_weight=self.reconstruction_loss_weight,
            name="autoencoder"
        )

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = int(np.prod(self._shape_before_bottleneck))
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_number}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_batch_normalization_{layer_number}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input,
                             [self.mu, self.log_variance, bottleneck],
                             name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_batch_normalization_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = x.shape[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)
        x = Sampling(name="encoder_output")([self.mu, self.log_variance])
        return x


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()