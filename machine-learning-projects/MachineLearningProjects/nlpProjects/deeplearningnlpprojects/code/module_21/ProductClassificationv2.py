# coding: utf-8
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100

EMBEDDING_FILE = "glove.6B.100d.txt"
category_index = {"clothing": 0, "camera": 1, "home-appliances": 2}
category_reverse_index = dict((y, x) for (x, y) in category_index.items())
STOPWORDS = set(stopwords.words("english"))

clothing = pd.read_csv("clothing.tsv", sep='\t')
cameras = pd.read_csv("cameras.tsv", sep='\t')
home_appliances = pd.read_csv("home.tsv", sep='\t')

datasets = [clothing, cameras, home_appliances]

print("Make sure there are no null values in the datasets")
for data in datasets:
    print("Has null values: ", data.isnull().values.any())


def preprocess(text):
    text = text.strip().lower().split()
    text = filter(lambda word: word not in STOPWORDS, text)
    return " ".join(text)


for dataset in datasets:
    dataset['title'] = dataset['title'].apply(preprocess)

all_texts = clothing['title'] + cameras['title'] + home_appliances['title']
all_texts = all_texts.drop_duplicates(keep=False)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_texts)

clothing_sequences = tokenizer.texts_to_sequences(clothing['title'])
electronics_sequences = tokenizer.texts_to_sequences(cameras['title'])
home_appliances_sequences = tokenizer.texts_to_sequences(home_appliances['title'])

clothing_data = pad_sequences(clothing_sequences, maxlen=MAX_SEQUENCE_LENGTH)
electronics_data = pad_sequences(electronics_sequences, maxlen=MAX_SEQUENCE_LENGTH)
home_appliances_data = pad_sequences(home_appliances_sequences, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
test_string = "sports action spy pen camera"
print("word\t\tid")
print("-" * 20)
for word in test_string.split():
    print("%s\t\t%s" % (word, word_index[word]))

test_sequence = tokenizer.texts_to_sequences(["sports action camera", "spy pen camera"])
padded_sequence = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
print("Text to Vector", test_sequence)
print("Padded Vector", padded_sequence)

print("clothing: \t\t", to_categorical(category_index["clothing"], 3))
print("camera: \t\t", to_categorical(category_index["camera"], 3))
print("home appliances: \t", to_categorical(category_index["home-appliances"], 3))

print("clothing shape: ", clothing_data.shape)
print("electronics shape: ", electronics_data.shape)
print("home appliances shape: ", home_appliances_data.shape)

data = np.vstack((clothing_data, electronics_data, home_appliances_data))
category = pd.concat([clothing['category'], cameras['category'], home_appliances['category']]).values
category = to_categorical(category)
print("-" * 10)
print("combined data shape: ", data.shape)
print("combined category/label shape: ", category.shape)

VALIDATION_SPLIT = 0.4
indices = np.arange(data.shape[0])  # get sequence of row index
np.random.shuffle(indices)  # shuffle the row indexes
data = data[indices]  # shuffle data/product-titles/x-axis
category = category[indices]  # shuffle labels/category/y-axis
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = category[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = category[-nb_validation_samples:]


from gensim.models import Word2Vec

#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE);
#print('Found %s word vectors of word2vec' % len(word2vec.vocab))

# odd man out
#print("Odd word out:", word2vec.doesnt_match("banana apple grapes carrot".split()))
#print("-" * 10)
#print("Cosine similarity between TV and HBO:", word2vec.similarity("tv", "hbo"))
#print("-" * 10)
#print("Most similar words to Computers:", ", ".join(map(lambda x: x[0], word2vec.most_similar("computers"))))
#print("-" * 10)

from keras.layers import Embedding

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
embeddings_index = dict()
f = open('glove.6B.100d.txt', mode='rt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

model = Sequential()
embedding_layer = Embedding(embedding_matrix.shape[0],  # or len(word_index) + 1
                            embedding_matrix.shape[1],  # or EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
model.add(embedding_layer)
model.add(Conv1D(filters=100, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize defined model
model.summary()

#300 dimensions .9995  # 100 dimensions .9997

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#example_product = "Nikon Coolpix A10 Point and Shoot Camera (Black)"
#example_product= "Red Tshirt"
example_product = "Air conditioner"

example_product = preprocess(example_product)
example_sequence = tokenizer.texts_to_sequences([example_product])
example_padded_sequence = pad_sequences(example_sequence, maxlen=MAX_SEQUENCE_LENGTH)

print("-" * 10)
print("Predicted category: ", category_reverse_index[model.predict_classes(example_padded_sequence, verbose=0)[0]])
print("-" * 10)
probabilities = model.predict(example_padded_sequence, verbose=0)
probabilities = probabilities[0]
print("Clothing Probability: ", probabilities[category_index["clothing"]])
print("Camera Probability: ", probabilities[category_index["camera"]])
print("home appliances probability: ", probabilities[category_index["home-appliances"]])
