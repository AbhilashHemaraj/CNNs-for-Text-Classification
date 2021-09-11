# Import Libraries and pretrained embeddings
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip

import os
import pandas as pd
import numpy as np
import string
import re
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Embedding


data_dir = '/content/drive/MyDrive/Datasets/'
train_dataset = pd.read_csv( data_dir + 'train.csv', index_col=0)
test_dataset = pd.read_csv( data_dir + 'test.csv', index_col=0)
val_dataset = pd.read_csv( data_dir + 'valid.csv', index_col=0)

glove_directory = data_dir + '/glove.6B.200d.txt'
glove_directory

"""# Data Preprocessing standard reference from keras: https://keras.io/examples/nlp/text_classification_from_scratch/"""
# Getting the dataset ready; converting pandas dataframe to tensor objects; converting them to batches of 64
dataset = tf.data.Dataset.from_tensor_slices((train_dataset.Text, train_dataset.Score))
train_dataset = dataset.shuffle(len(train_dataset)).batch(64)

dataset = tf.data.Dataset.from_tensor_slices((test_dataset.Text, test_dataset.Score))
test_dataset = dataset.shuffle(len(test_dataset)).batch(64)

dataset = tf.data.Dataset.from_tensor_slices((val_dataset.Text, val_dataset.Score))
val_dataset = dataset.shuffle(len(val_dataset)).batch(64)

embeddings_index = {}
with open(glove_directory, encoding="utf8") as f:
    for i in tqdm(f):
        word, vectors = i.split(maxsplit=1)
        vectors = np.fromstring(vectors, "f", sep=" ")
        embeddings_index[word] = vectors

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )

def plot_history(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'b', label='Training acc')
    plt.plot(x, val_accuracy, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

# Model constants.
max_features = 20000
embedding_dim = 200 # use 300 and rerun fro phase-1 experiments
sequence_length = 500


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

text_ds = train_dataset.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the data.
train_ds = train_dataset.map(vectorize_text)
val_ds = val_dataset.map(vectorize_text)
test_ds = test_dataset.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

vocab = vectorize_layer.get_vocabulary()
num_tokens = len(vocab) + 2
word_index = dict(zip(vocab, range(len(vocab))))

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True,
)

"""# Model with pretrained embeddings"""
# New variation of above model
inputs = keras.Input(shape=(None,), dtype="int32")
embeddings = embedding_layer(inputs)
x = Conv1D(256, 7, activation="relu")(embeddings)
x = MaxPooling1D()(x)
x = Conv1D(128, 5, activation="relu")(x)
x = MaxPooling1D()(x)
x = Conv1D(128, 5, activation="relu")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(6, activation="softmax")(x)
model = keras.Model(inputs, preds)
model.summary()
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

epochs = 5

# Fit the model using the train and test datasets.
history_pretrained = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

history_pretrained.model.save(data_dir + "pretrained_single_channel")


###############################################################################
"""# Model without pretrained embeddings"""

# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Let's make a text-only dataset (no labels):
text_ds = train_dataset.map(lambda x, y: x)

# Let's call `adapt`:
vectorize_layer.adapt(text_ds)

# Vectorize the data.
train_ds = train_dataset.map(vectorize_text)
val_ds = val_dataset.map(vectorize_text)
test_ds = test_dataset.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)


inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = Embedding(max_features, embedding_dim)(inputs)
x = Dropout(0.5)(x)
x = Conv1D(512, 7, padding="valid", activation="relu", strides=5)(x)
x = Conv1D(256, 7, padding="valid", activation="relu", strides=3)(x)
x = Conv1D(128, 7, padding="valid", activation="relu", strides=4)(x)
x = Conv1D(64, 7, padding="valid", activation="relu", strides=4)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(6, activation="softmax")(x)
model = keras.Model(inputs, preds)
model.summary()
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

epochs = 5

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
history.model.save(data_dir + "conv_512_relu_80")

plot_history(history)

"""# Testing above model using Swish Activation function"""

# Importing the sigmoid function from
# Keras backend and using it
from keras.backend import sigmoid

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))


# Getting the Custom object and updating them
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = Embedding(max_features, embedding_dim)(inputs)
x = Dropout(0.5)(x)
x = Conv1D(512, 7, padding="valid", activation="swish", strides=5)(x)
x = Conv1D(256, 7, padding="valid", activation="swish", strides=3)(x)
x = Conv1D(128, 7, padding="valid", activation="swish", strides=4)(x)
x = Conv1D(64, 7, padding="valid", activation="swish", strides=4)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation="softmax")(x)
model1 = keras.Model(inputs, preds)
model1.summary()
model1.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

epochs = 5

# Fit the model using the train and test datasets.
history_1 = model1.fit(train_ds, validation_data=val_ds, epochs=epochs)

plot_history(history_1)


"""# Multi-Channel CNNs"""
# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

# channel 1
inputs = tf.keras.Input(shape=(None,), dtype="int64")
embedding_1 = x = Embedding(max_features, embedding_dim)(inputs)
convolution_1 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(embedding_1)
convolution_1 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(convolution_1)
pooling_1 = GlobalMaxPooling1D()(convolution_1)
flatten_1 = Flatten()(pooling_1)

# channel 2
embedding_2 = x = Embedding(max_features, embedding_dim)(inputs)
convolution_2 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(embedding_2)
convolution_2 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(convolution_2)
pooling_2 = GlobalMaxPooling1D()(convolution_2)
flatten_2 = Flatten()(pooling_2)

# channel 3
embedding_3 = x = Embedding(max_features, embedding_dim)(inputs)
convolution_3 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(embedding_3)
convolution_3 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(convolution_3)
pooling_3 = GlobalMaxPooling1D()(convolution_3)
flatten_3 = Flatten()(pooling_3)

concatenated_features = concatenate([flatten_1, flatten_2, flatten_3])
x = Dropout(0.5)(concatenated_features)
preds = Dense(6, activation="softmax")(x)
multi_model = keras.Model(inputs = inputs, outputs = preds)
multi_model.summary()

tf.keras.utils.plot_model(multi_model, to_file='multi_model_1.png', show_shapes=True)

## just testing different variations
# channel 1
inputs = tf.keras.Input(shape=(None,), dtype="int64")
embedding_1 = x = Embedding(max_features, embedding_dim)(inputs)
convolution_1 = Conv1D(128, 5, padding="valid", activation="relu", strides=1)(embedding_1)
convolution_1 = Conv1D(128, 5, padding="valid", activation="relu", strides=1)(convolution_1)
pooling_1 = GlobalMaxPooling1D()(convolution_1)
flatten_1 = Flatten()(pooling_1)

# channel 2
embedding_2 = x = Embedding(max_features, embedding_dim)(inputs)
convolution_2 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(embedding_2)
convolution_2 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(convolution_2)
pooling_2 = GlobalMaxPooling1D()(convolution_2)
flatten_2 = Flatten()(pooling_2)

# channel 3
embedding_3 = x = Embedding(max_features, embedding_dim)(inputs)
convolution_3 = Conv1D(128, 3, padding="valid", activation="relu", strides=2)(embedding_3)
convolution_3 = Conv1D(128, 3, padding="valid", activation="relu", strides=2)(convolution_3)
pooling_3 = GlobalMaxPooling1D()(convolution_3)
flatten_3 = Flatten()(pooling_3)

concatenated_features = concatenate([flatten_1, flatten_2, flatten_3])
x = Dropout(0.5)(concatenated_features)
preds = Dense(6, activation="softmax")(x)
multi_model = keras.Model(inputs = inputs, outputs = preds)
multi_model.summary()
