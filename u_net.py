# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:11:49 2019

@author: ma10s
"""
from tensorflow import keras

import matplotlib.pyplot as plt


# Class definition
class Unet_model():
    """
    We use keras to define CNN and DNN layers to the model
    """

    def __init__(self, input_shape=(420,580,1), output_shape=15, con_len=3, con_layers=[25, 50, 100, 200],
                 last_pooling=keras.layers.AvgPool1D, dense_layers=[100, 100], dataname='noname'):
        self.name = 'unet_con_len' + str(con_len) + '_con_layers' + str(con_layers) + '_dense_layers' + str(
            dense_layers) + '_data' + dataname

        self.model = construct_model(input_shape, output_shape, con_len=con_len, con_layers=con_layers,
                                     last_pooling=last_pooling, dense_layers=dense_layers)
        self.save_as = 'saved_models/' + self.name

    # train the model given the data
    def train(self, inputs, targets, validation_inputs=None, validation_targets=None, batch_size=32, epochs=20, learning_rate=0.001,
              save_model=True, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=1):

        es = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=verbose,
                                           patience=early_stopping_patience)

        if save_model:
            mcp_save = keras.callbacks.ModelCheckpoint(self.save_as + '.hdf5',
                                                       save_best_only=True,
                                                       monitor='accuracy',
                                                       mode='min')
        # Using Adam optimizer
        
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        if validation_inputs is not None:
            history = self.model.fit(
                inputs, targets, validation_data=(validation_inputs,
                                                  validation_targets), epochs=epochs, batch_size=batch_size, shuffle=True,
                callbacks=[mcp_save, es], verbose=verbose)
        else:
            history = self.model.fit(
                inputs, targets, epochs=epochs, batch_size=batch_size,
                shuffle=True,
                callbacks=[mcp_save, es], verbose=verbose)


        # To avoid overfitting load the model with best validation results after
        # the first training part.
        if save_model:
            self.model = keras.models.load_model(self.save_as + '.hdf5')

        # TODO: concatenate history1 and history2 to plot all the training
        # progress
        if plot_training_progress:
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])

        return history

    # Predict
    def predict(self, xt):
        # predict
        return self.model.predict(xt)

    def load_model(self, save_as=None):
        if save_as is None:
            save_as = self.save_as
        print("model load: ", save_as)
        print("self.name: ", self.name, ", self.save_as: ", self.save_as)
        self.model = keras.models.load_model(save_as + '.hdf5')


def construct_model(input_shape, output_shape, con_len=3, con_layers=[25, 50, 100], last_pooling=keras.layers.AvgPool2D,
                    dense_layers=[100, 100]):
    # TODO: add a **kwargs to specify the hyperparameters
    activation = 'relu'
    dense_activation = 'relu'
    padding = 'same'
    poolpadding = 'valid'

    maxpool = con_len
    levels = 3
    batch_mom = 0.99
    reg = None
    # pool = keras.layers.AvgPool1D #
    pool = keras.layers.MaxPooling2D
    model = keras.Sequential()
    depth = input_shape[0]
    levels = len(con_layers) - 1
    inputs = keras.Input(input_shape)
    conv = []
    pool = []
    conv_ = inputs
    for level in range(levels):
        conv_ = keras.layers.Conv2D(con_layers[level], con_len, activation='relu', padding='same', kernel_initializer='he_normal')(conv_)
        conv_ = keras.layers.Conv2D(con_layers[level], con_len, activation='relu', padding='same', kernel_initializer='he_normal')(conv_)
        conv.append(conv_)
        conv_ = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_)
        pool.append(conv_)

    conv_ = keras.layers.Conv2D(con_layers[levels], con_len, activation='relu', padding='same',
                                kernel_initializer='he_normal')(conv_)
    conv_ = keras.layers.Conv2D(con_layers[levels], con_len, activation='relu', padding='same',
                                kernel_initializer='he_normal')(conv_)

    for level in range(levels-1,-1,-1):
        conv_ = keras.layers.Conv2D(con_layers[level], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_))

        print("conv_ shape: ", conv_.input_shape, " - ", conv_.output_shape)
        conv_ = keras.layers.concatenate([conv[level], conv_], axis=3)
        conv_ = keras.layers.Conv2D(con_layers[level], con_len, activation='relu', padding='same', kernel_initializer='he_normal')(conv_)
        conv_ = keras.layers.Conv2D(con_layers[level], con_len, activation='relu', padding='same', kernel_initializer='he_normal')(conv_)

    conv_ = keras.layers.Conv2D(con_layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_)
    conv_ = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_)
    conv_ = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv_)
    model = keras.models.Model(inputs=inputs, outputs=conv_)

    model.summary()
    return model


