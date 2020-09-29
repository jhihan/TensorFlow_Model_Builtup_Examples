import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from sklearn import preprocessing
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def model_seq(max_length, vocab_size, embedding_dim, n_labels, embedding_matrix=None):
    """The bidirectional LSTM model builded up with sequential API
    :param max_length: (int) The maximal length of the training sequence
    :param vocab_size: (int) The maximum number of words to keep, based on word frequency.
    :param embedding_dim: (int) The dimension of the embedding vector.
    :param n_labels: (int) Number of the labels
    :return: Bidirectional LSTM model builded up with sequential API
    """
    if embedding_matrix == None:
      model1 = tf.keras.Sequential([
               tf.keras.layers.Embedding( input_dim = vocab_size, output_dim = embedding_dim),
        # If we want to have flatten layer, we must assign input_length = max_length in the Embedding layer.
        # Here Flatten is not needed because we want to use RNNs - model.
               tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(embedding_dim) ),
               tf.keras.layers.Dense( embedding_dim, activation='relu' ),
               tf.keras.layers.Dense( n_labels, activation='softmax' )
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
           ])
    else:
        model1 = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                      embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                                      trainable = False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(n_labels, activation='softmax')
        ])
    return model1


def model_fun(max_length, vocab_size, embedding_dim, n_labels, embedding_matrix=None):
    """The bidirectional LSTM model builded up with functional API
    :param max_length: (int) The maximal length of the training sequence
    :param vocab_size: (int) The maximum number of words to keep, based on word frequency.
    :param embedding_dim: (int) The dimension of the embedding vector.
    :param n_labels: (int) Number of the labels
    :return: Bidirectional LSTM model builded up with functional API
    """
    inputs = tf.keras.Input(shape=(max_length,))

    if embedding_matrix == None:
        embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                              input_length=max_length)(inputs)
    else:
        embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length,
                                            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                            trainable=False)(inputs)

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim))(embedding)
    dense1 = tf.keras.layers.Dense(embedding_dim, activation='relu')(bi_lstm)
    outputs = tf.keras.layers.Dense(n_labels, activation='softmax')(dense1)
    model2 = tf.keras.Model(inputs=inputs, outputs=outputs, name="fun_model")
    return model2


class Model_sub(tf.keras.models.Model):
    """The bidirectional LSTM model builded up via subclassing
    """
    def __init__(self,max_length, vocab_size, embedding_dim, n_labels, embedding_matrix=None):
        super(Model_sub, self).__init__()
        self.max_length = max_length
        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                                       input_length=max_length)
        else:
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                                       input_length=max_length,
                                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                       trainable=False)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim))
        self.dense1 = tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))
        self.output_layer= tf.keras.layers.Dense(n_labels, activation='softmax')

    def call(self, x, training=None):
        # Some layers, in particular the BatchNormalization layer and the Dropout layer, have different behaviors
        # during training and inference. For such layers, it is standard practice to expose a training (boolean)
        # argument in the call() method.
        # Example:
        # self.dropout = tf.keras.layers.Dropout(0.5)
        # if training:
        #   x = self.dropout(x, training=training)

        x1 = self.embedding(x)
        x2 = self.biLSTM(x1)
        x2 = self.dense1(x2)
        if (training != False):
            x2 = self.dropout(x2, training=training)
        x3 = self.dense2(x2)
        output = self.output_layer(x3)
        return output

    def summary(self):
        """
        This method overrides the summary() method in tf.keras.models.Model
        :return: The model summary is print out
        """
        x = tf.keras.Input(shape=(self.max_length,))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x)).summary()


if __name__ == "__main__":

    print(tf.__version__)
    STOPWORDS = set(stopwords.words('English'))

    #-- set up hypterparameter/parameter
    vocab_size = 5000
    embedding_dim = 100
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    training_portion = .8


    is_pretrained = True
    if is_pretrained == True:
        assert embedding_dim in [50,100,200,300], "embedding_dim must be 50,100,200 or 300"
        if (embedding_dim == 50):
            path_to_glove_file = os.path.join(
                os.path.expanduser("../"), "embeddings/glove.6B/glove.6B.50d.txt"
              )
        elif (embedding_dim == 100):
            path_to_glove_file = os.path.join(
                os.path.expanduser("../"), "embeddings/glove.6B/glove.6B.100d.txt"
              )
        elif (embedding_dim == 200):
            path_to_glove_file = os.path.join(
                os.path.expanduser("../"), "embeddings/glove.6B/glove.6B.200d.txt"
              )
        elif (embedding_dim == 200):
            path_to_glove_file = os.path.join(
                os.path.expanduser("../"), "embeddings/glove.6B/glove.6B.300d.txt"
              )

    articles = []
    labels = []

    with open("bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            article = row[1]
            for word in STOPWORDS:
                token = ' '+ word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ',' ')
            articles.append(article)

    print("Number of dataset:",len(articles))
    print("Number of dataset (label):",len(labels))

    train_size = int( len(articles) * training_portion )

    train_articles = articles[:train_size]
    train_labels = labels[:train_size]
    test_articles = articles[train_size:]
    test_labels = labels[train_size:]


    print("Training dataset size:",train_size)
    print(len(train_articles))
    print(len(train_labels))
    print("Testing dataset size:",len(test_articles))
    print(len(test_labels))


# ---- Preprocessing: label to index -------
    le = preprocessing.LabelEncoder()
    le.fit( labels )
    label_train_seq = np.array( le.transform( train_labels ) )
    label_test_seq = np.array( le.transform( test_labels ) )
    n_labels = len( le.classes_ )

    label_index = {}
    for index,item in enumerate(le.classes_):
        label_index[item] = index

    print( label_train_seq[:10] )
    print( label_test_seq[:10] )
    print( "length of the labels:" , n_labels )
    print( le.classes_ )
    print(label_index)
# ---------------------------------

# ---- Preprocessing: text to index sequence --------
# TextVectorization is another choice.

    tokenizer = Tokenizer( num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts( train_articles )
    word_index = tokenizer.word_index
    print ( dict( list( word_index.items() )[0:10] ) )

    train_sequences = tokenizer.texts_to_sequences( train_articles )
# Notice that only the most common ( num_words-1 ) words will be kept

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type )

    print("Size of word_index %s " % len( list( word_index.items() ) ) )
    print("vocab_size= %s" % vocab_size )
    print("Minimal index of train_sequences %s" % min( [item for sublist in train_sequences for item in sublist] ))
    print("Maximal index of train_sequences %s" % max( [item for sublist in train_sequences for item in sublist] ))
    print("Minimal index of train_padded %s" % min( [item for sublist in train_padded for item in sublist] ))
    print("Maximal index of train_padded %s" % max( [item for sublist in train_padded for item in sublist] ))



    test_sequences = tokenizer.texts_to_sequences( test_articles )
    test_padded = pad_sequences( test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type )

# ---- Load pre-trained word embeddings ---
    if is_pretrained == True:

        embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        print("Found %s word vectors." % len(embeddings_index))

        num_tokens = vocab_size # Here padding:0 and <VVO>:1 are also included
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word in list(word_index)[:num_tokens-1]:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                i = word_index[word]
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
    else:
        print("The Embedding layer has not been trained yet.")
#-----------------------------------------


    model = Model_sub(max_length, vocab_size, embedding_dim, n_labels, embedding_matrix )
    #model = model_seq(max_length, vocab_size, embedding_dim, n_labels)
    #model = model_fun(max_length, vocab_size, embedding_dim, n_labels)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # categorical_crossentropy (cce) uses a one-hot array to calculate the probability,
    # sparse_categorical_crossentropy (scce) uses a category index

    num_epochs = 40
    history = model.fit(train_padded, label_train_seq, epochs=num_epochs, validation_split= 0.2 , batch_size=200, verbose=2)
    #If we want to use the test dataset as the validation data:
    #history = model.fit(train_padded, label_train_seq, epochs=num_epochs, validation_data=(test_padded, label_test_seq), batch_size=200, verbose=2)

    scores = model.evaluate( test_padded, label_test_seq )
    print('accuracy=', scores[1])

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")