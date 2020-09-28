# 3 ways to build a neural network model in Keras with TensorFlow (Sequential, Functional, and Model Subclassing)
In this project, I present the three ways to build up the model in TensorFlow.keras: Sequential, functional, and model subclassing.
Here I will solve a BBC news document classification problem with the data set using bidirectional LSTM model as an example.
## Dataset
The BBC news classification 
https://raw.githubusercontent.com/jhihan/TensorFlow_Model_Builtup_Examples/master/bbc-text.csv
The GloVe (Global Vectors for Word Representation)
http://nlp.stanford.edu/data/glove.6B.zip
## Model
### Sequential API
### Functional API
### Model Subclassing
Model subclassing is harder to utilize than the Sequential or Functional. Actually, we don't need model subclassing in this problem. But this mothod has flexible for us to control  every nuance of the network and training process.

```
class Model_sub(tf.keras.models.Model):
  def __init__(self):
    pass
  def call(self, x, training=None):
  # The argument training is needed only if the layers which have different behaviors during training and inference are considered.
    pass
  def summary(self):
  # This method is only needed if we want to check the model summary before compile.
    pass
```
