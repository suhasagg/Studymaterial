# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
#Below optimisation -
#embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if word in word2vec.vocab:
#        embedding_matrix[i] = word2vec.word_vec(word)
#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
#Optimisation of embedding layers using different embedding variants
#1)sentence transformer based (twitter,wikipedia)
#2)word2vec (twitter,wikipedia,conceptnet)
#embedding_layer = Embedding(embedding_matrix.shape[0],  # or len(word_index) + 1
#                            embedding_matrix.shape[1],  # or EMBEDDING_DIM,
#
#
#
#
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
