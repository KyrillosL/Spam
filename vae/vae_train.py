'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.layers import Lambda, Input, Dense, LSTM, RepeatVector, Conv2D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K, objectives
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import pretty_midi
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

intermediate_dim = 128
batch_size = 8
latent_dim = 2
epochs = 1000
random_state = 42

list_files_name= []
file_shuffle=[]
test_size=0.25


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    #print(len(z_mean))

    #to_decode = np.array([[0.5, 0], [1.8, 1]], dtype=np.float32)
    #final = decoder.predict(to_decode)
    #print(final )

   # print("ICI ", file_shuffle[:int(dataset_size * test_size)])
    #for i, txt in enumerate(file_shuffle[ :int(dataset_size*2 * test_size)]):
        #print("i ", i)
        #plt.annotate(txt,(z_mean[i,0], z_mean[i,1]))

    plt.show()


def load_data(path ):


    metadata = pd.read_csv(path)
    # Iterate through each midi file and extract the features

    i=0
    for index, row in metadata.iterrows():

        if i<4000:
            class_label = row["is_spam"]
            #print("row", row)
            data = []
            for x in row[1:-1]:
                x = round(x,2)
                #print("X", x)
                data.append(x)
            features.append([data, class_label])
            i+=1


print("LOADING DATA FOR TRAINING...")
features = []

path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/fouille_de_données/Projet/dataset.csv"
load_data(path_to_load)


# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' mails')

# Convert features & labels into numpy arrays
listed_feature = featuresdf.feature.tolist()

X = np.array(listed_feature).astype(dtype=object)
y = np.array(featuresdf.class_label.tolist())

print("Shape: ", X.shape, y.shape)
# split the dataset


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
X_shuffle = shuffle(X, random_state=random_state)
y_shuffle = shuffle(y, random_state=random_state)
file_shuffle = shuffle(list_files_name, random_state=random_state)



midi_file_size = x_train.shape[1]
original_dim = midi_file_size
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 100
x_test = x_test.astype('float32') / 100

# network parameters
input_shape = (original_dim, )

#ORIGINAL MODEL.
inputs = Input(shape=input_shape, name='encoder_input')

# VAE model = encoder + decoder
# build encoder model

x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()




# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


if __name__ == '__main__':

    models = (encoder, decoder)
    data = (x_test, y_test)


    reconstruction_loss = mse(inputs, outputs)
    #reconstruction_loss = binary_crossentropy(inputs,outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)


    vae.add_loss(vae_loss)
    opt = Adam(lr=0.0001)  # 0.001 was the default, so try a smaller one
    vae.compile(optimizer=opt,  metrics=['accuracy'])
    vae.summary()



    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    # train the autoencoder

    score=vae.fit(x_train,
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            validation_data=(x_test, None),
            callbacks=[es])
    vae.save_weights('vae_mlp_mnist.h5')
    

    score2 = vae.evaluate(x_test, None, verbose=1)
    print('Score', score.history)
    print('Score', score2)
    plot_results(models, data)
    
    print("LOADING ALL EMAILS")
    data = (X, y)
    
    plot_results(models, data)
