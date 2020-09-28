# 3 ways to build a neural network model in Keras with TensorFlow (Sequential, Functional, and Model Subclassing)
In this project, I present the three ways to build up the model in TensorFlow.keras: Sequential, functional, and model subclassing.
Here I will solve a BBC news document classification problem with the data set using bidirectional LSTM model as an example.
## Dataset
The BBC news classification:  
https://raw.githubusercontent.com/jhihan/TensorFlow_Model_Builtup_Examples/master/bbc-text.csv  
The GloVe (Global Vectors for Word Representation)  
http://nlp.stanford.edu/data/glove.6B.zip  
## Model
```
import tensorflow as tf
```
### Sequential API
```
model1 = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=100,input_length=200),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu')
            tf.keras.layers.Dense(5, activation='softmax')
        ])
```
### Functional API
In the functional API, models can be created with non-linear topology and are not obliged to follow a straight line like in the sequential model.
```
inputs = tf.keras.Input(shape=(200,))
embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=100,input_length=200)(inputs)
bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim))(embedding)
dense1 = tf.keras.layers.Dense(embedding_dim, activation='relu')(bi_lstm)
outputs = tf.keras.layers.Dense(n_labels, activation='softmax')(dense1)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fun_model")
```
### Model Subclassing
Model subclassing is harder to utilize than the Sequential or Functional. Actually, we don't need model subclassing in this problem. But this mothod has flexible for us to control every nuance of the network and training process. The template of the model subclassing can be represented as following:

```
class Model_sub(tf.keras.models.Model):
    def __init__(self):
        super(Model_sub, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=100,input_length=200)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.output_layer= tf.keras.layers.Dense(5, activation='softmax')
    def call(self, x, training=None):
    # The argument training is needed only if the layers which have different behaviors during training and inference are considered.
        x = self.embedding(x)
        x = self.biLSTM(x)
        x = self.dense1(x)
        if (training != False):
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output
    def summary(self):
    # This method is only needed if we want to check the model summary before compile.
        x = tf.keras.Input(shape=(200,))
            return tf.keras.models.Model(inputs=[x], outputs=self.call(x)).summary() 
```
We can build up a neural network template by inheriting the tf.keras.models.Model and overriding two (or more) functions. 
#### Overriding ```__init__```
The first function is the constructor ```__init__```, in which all the layers can be set up. 
#### Overriding ```call```
The second function is ```call```, in which the feedforward in the neural network is executed. That means the data will flow from the first layer to other layers and finally out of the outlayer to complete a neural network operation. Some layers, in particular the BatchNormalization layer and the Dropout layer, have different behaviors during training and inference. For such layers, it is standard practice to expose a training (boolean) argument in the call() method.
#### Overriding or defining other functions if needed
Here the third function "summary" normally is not necessary. If we want to print out the model summary before the compile process, the model must be build up first. There are several way to achieve this purpose and overriding the summary function as above is one of the way. If you are interested in the model summary in the model subclassing method, please check the discussion in this post: https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model.
