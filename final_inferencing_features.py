import keras as k
import keras.backend as K
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
import cv2
import time

#load image data
all_data = np.load('images_dataset.npy')

###Just in case you want to try for few images, index all_data accordingly
X_test = all_data
print(X_test.shape)

def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s

from keras.models import load_model
s = reset_tf_session()

##Load saved model. Also spend time in tuning the model. This is a generic model we created from the training script.
autoencoder = load_model("hackathon_autoencoder.86-479.35.h5")  
encoder = autoencoder.layers[1]
decoder = autoencoder.layers[2]



def generate_embedding(img,encoder,decoder):
    """Inference the model to generate embedding"""
    start = time.time()
    code = encoder.predict(img[None])[0]  # Generated Embedding
    end = time.time()
    #print(end - start)
    reconstructed = decoder.predict(code[None])[0]
    
    return reconstructed, code

all_embeddings = list()
for i,row in enumerate(X_test):
    reconstruct, code = generate_embedding(row, encoder, decoder)
    all_embeddings.append(code)
embeddings_array = np.stack(all_embeddings)


#### Save image embedding to use for recommendation. you have to save embeddings_array (list of list) to use it later.  These are 128 lenght vector embedding
