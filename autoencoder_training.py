import keras as k
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.callbacks import History
from keras.layers import Activation
from keras.models import model_from_json
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.ndimage import rotate as rot
from sklearn.model_selection import train_test_split
from keras import utils
import numpy as np
import tensorflow as tf


def my_autoencode(img_shape, code_size=32):
    H,W,C = img_shape
    
    # encoder
    encoder = k.models.Sequential()
    encoder.add(k.layers.InputLayer(img_shape))
    encoder.add(k.layers.Conv2D(32, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.AveragePooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Flatten())
    encoder.add(k.layers.Dense(512, activation='elu'))
    encoder.add(k.layers.Dense(256, activation='elu'))
    encoder.add(k.layers.Dense(code_size, activation='elu'))
    encoder.summary()

    # decoder
    decoder = k.models.Sequential()
    decoder.add(k.layers.InputLayer((code_size,)))
    decoder.add(k.layers.Dense(256, activation='elu'))
    decoder.add(k.layers.Dense(512, activation='elu'))
    decoder.add(k.layers.Dense(8192, activation='elu'))
    decoder.add(k.layers.Reshape((8, 8, 128)))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(64, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(64, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(32, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='elu', padding='same')) # Unsure about this
    decoder.summary()
    
    return encoder, decoder

all_data = np.load('images_dataset.npy')
X_train = all_data[:2723, :]
X_test = all_data[2723:, ]
shape = X_train[0].shape # Get from dataset
encoder, decoder = my_autoencode(shape, code_size=128)
inp = k.layers.Input(shape)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = k.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('hackathon_autoencoder.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True)
]


# If you want to resume from a checkpoint
#     import keras.backend as K
#     def reset_tf_session():
#         K.clear_session()
#         tf.reset_default_graph()
#         s = K.get_session()
#         return s
#     #### uncomment below to continue training from model checkpoint
#     #### every time epoch counter starts at 0, so you need to track epochs manually
#     from keras.models import load_model
#     s = reset_tf_session()
#     autoencoder = load_model("checkpoints/hackathon_autoencoder.78-508.84.h5")  # continue after epoch 0+1
#     encoder = autoencoder.layers[1]
#     decoder = autoencoder.layers[2]

# # Train Model
autoencoder.fit(x=X_train, y=X_train,
                validation_data=[X_test, X_test],
                epochs=200,
                batch_size=32,
                shuffle=True,
                callbacks = callbacks
               )