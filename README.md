# Capillary Helper Scripts

ALS.py : collaborative filtering based recommendation script

Image embedding generation:
For image embedding generation following scripts will be useful -

1. autoencoder_training.py: Training script of autoencoder. saves the model

2. hackathon_autoencoder.86-479.35.h5 : Basic autoencoder model based on the script autoencoder_training.py (can be tuned further)

3. image_reader_converter.py : Created input dataset from given images , suitable to feed to autoencoder for embedding generation

4. final_inferencing_features.py : inferencing script, loads the autoencoder model and generates 128 lenght vector for each image
