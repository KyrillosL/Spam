import pretty_midi
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation, Dropout
from keras.callbacks import EarlyStopping


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


filepath = "/Users/Cyril_Musique/Documents/Cours/M2/fouille_de_données/Projet/dataset.csv"
metadata = pd.read_csv(filepath)

features = []

# Iterate through each midi file and extract the features
for index, row in metadata.iterrows():
    class_label = float(row["is_spam"])/100
    data = []
    for x in row[1:-1]:
        x = round(x,2)
        #print("X", x)
        data.append(x)
    features.append([data, class_label])


# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')


# Convert features & labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

num_labels = yy.shape[1]
filter_size = 2

# split the dataset
from sklearn.model_selection import train_test_split

print(X.shape,y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)



#Model
model = Sequential()

model.add(Dense(256, input_shape=(57,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 100
num_batch_size = 16

checkpointer = ModelCheckpoint(filepath='weights.best.basic_mlp.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])



def extract_feature(file_name):

    return np.array([0])

def print_prediction(file_name):
    prediction_feature = extract_feature(file_name)
    predicted_vector = model.predict(prediction_feature)
    print(predicted_vector)

#file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100/generated_8.mid'
#print_prediction(file_to_test)
