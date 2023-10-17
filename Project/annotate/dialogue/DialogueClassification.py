from google.colab import drive
drive.mount('/content/drive')

!pip install kashgari
!pip install tensorflow_addons==0.13.0
!pip install tensorflow==2.3.0

import datetime
import os
import re
import tensorflow as tf
print(tf.__version__)
import kashgari
from kashgari.tokenizers import BertTokenizer
from kashgari.embeddings import BareEmbedding, BertEmbedding
from kashgari.tasks.classification import CNN_Model
from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel
from tensorflow import keras
import tensorflow.keras.backend as K
from typing import Dict, Any
import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

pre_trained = 'roberta'
file_path = "/content/drive/MyDrive/ColabNotebooks/project_data/data.txt"
bert_path = "/content/drive/MyDrive/ColabNotebooks/roberta_zh/resources/"


def load_data(filepath, bertpath):
  X, y = [], []
  x_train, x_valid, x_test, y_train, y_valid, y_test = [], [], [], [], [], []
  label_list = []
  label_counter = {}

  global embed, tokenizer
  embed = BertEmbedding(bertpath)
  tokenizer = BertTokenizer.load_from_vocab_file(os.path.join(bert_path, 'vocab.txt'))

  pattern = '[，、。:：；/（）()《》“”"？,.;?·…0-9A-Za-z+=-]'
  lines = open(filepath, 'r', encoding='utf-8').read().splitlines()
  for line in tqdm.tqdm(lines):
    rows = line.split('\t')
    if len(rows) == 4:
      content = tokenizer.tokenize(re.sub(pattern, "", rows[0]))
      label = rows[1]
      X.append(content)
      y.append(label)
      if label not in label_list:
        label_list.append(label)
        label_counter[label] = 1
      else:
        label_counter[label] += 1
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,shuffle=True)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=0.25,shuffle=True)

  print("Project Data Summary:")
  print("Train\t", len(x_train))
  print("Valid\t", len(x_valid))
  print("Test\t", len(x_test))
  print("Label\t", len(label_list))
  for key, value in label_counter.items():
    print(key + '\t', value)
  return x_train, y_train, x_test, y_test, x_valid, y_valid ,embed

Xtr, ytr, Xts, yts, Xva, yva, bert_embedding = load_data(file_path, bert_path)

# Run CNN_Model with RoBERTa
tf_board = TensorBoard(log_dir='tf_dir/cnn_model',
                       histogram_freq=5,
                       update_freq='batch')
K.clear_session()
model = CNN_Model(bert_embedding)
model.fit(Xtr, ytr, Xva, yva,callbacks=[tf_board],epochs=30,batch_size=32)
report = model.evaluate(Xts, yts)

%load_ext tensorboard
%tensorboard --logdir tf_dir/cnn_model

class MY_CNN_BILSTM_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'bilstm_0': {
                'units': 64,
                'return_sequences': True
            },
            'conv_0': {
                'filters': 32,
                'kernel_size': 5,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
            },
            'concat': {
                'axis': 1
            },
            'dropout': {
                'rate': 0.5
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.label_processor.vocab2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # One 1D Convolutional layer and one BiLSTM layer
        layers_rcnn_seq = []
        layers_rcnn_seq.append(L.Conv1D(**config['conv_0']))
        layers_rcnn_seq.append(L.Bidirectional(L.LSTM(**config['bilstm_0'])))

        # Max Pooling
        layers_sensor = []
        layers_sensor.append(L.GlobalMaxPooling1D())
        layer_concat = L.Concatenate(**config['concat'])

        # Two fully connected layers
        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_rcnn_seq:
            tensor = layer(tensor)

        tensor_output = layers_sensor[0](tensor)

        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = keras.Model(embed_model.inputs, tensor_output)


# Run MY_CNN_BILSTM_Model with RoBERTa
tf_board = TensorBoard(log_dir='tf_dir/cnn_bilstm_model',
                       histogram_freq=5,
                       update_freq='batch')
K.clear_session()
model = MY_CNN_BILSTM_Model(bert_embedding)
model.fit(Xtr, ytr, Xva, yva,callbacks=[tf_board],epochs=30,batch_size=32)
report = model.evaluate(Xts, yts)

%load_ext tensorboard
%tensorboard --logdir tf_dir/cnn_bilstm_model

class MY_Double_BILSTM(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.2
            },
            'bilstm_0': {
                'units': 64,
                'return_sequences': True
            },
            'bilstm_1': {
                'units': 64,
                'return_sequences': True
            },
            'conv_0': {
                'filters': 32,
                'kernel_size': 5,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
            },
            'concat': {
                'axis': 1
            },
            'dropout': {
                'rate': 0.5
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.label_processor.vocab2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Two BiLSTM Layers
        layers_rcnn_seq = []
        layers_rcnn_seq.append(L.Bidirectional(L.LSTM(**config['bilstm_0'])))
        layers_rcnn_seq.append(L.Bidirectional(L.LSTM(**config['bilstm_1'])))

        # Max Pooling
        layers_sensor = []
        layers_sensor.append(L.GlobalMaxPooling1D())
        # layers_sensor.append(L.GlobalAveragePooling1D())
        layer_concat = L.Concatenate(**config['concat'])

        # Two fully connected layers
        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_rcnn_seq:
            tensor = layer(tensor)

        tensor_output = layers_sensor[0](tensor)

        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = keras.Model(embed_model.inputs, tensor_output)


# Run MY_Double_BILSTM model with RoBERTa
tf_board = TensorBoard(log_dir='tf_dir/double_bilstm_model',
                       histogram_freq=5,
                       update_freq='batch')
K.clear_session()
model = MY_Double_BILSTM(bert_embedding)
model.fit(Xtr, ytr, Xva, yva,callbacks=[tf_board],epochs=30,batch_size=32)
report = model.evaluate(Xts, yts)
print(report)

%load_ext tensorboard
%tensorboard --logdir tf_dir/double_bilstm_model

class My_Double_CNN(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.2
            },
            'bilstm_0': {
                'units': 64,
                'return_sequences': True
            },
            'conv_0': {
                'filters': 128,
                'kernel_size': 5,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
            },
            'conv_1': {
                'filters': 32,
                'kernel_size': 5,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
            },
            'concat': {
                'axis': 1
            },
            'dropout': {
                'rate': 0.5
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.label_processor.vocab2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Two Convolution Layers
        layers_rcnn_seq = []
        layers_rcnn_seq.append(L.Conv1D(**config['conv_0']))
        layers_rcnn_seq.append(L.Conv1D(**config['conv_1']))

        # Max Pooling
        layers_sensor = []
        layers_sensor.append(L.GlobalMaxPooling1D())
        layer_concat = L.Concatenate(**config['concat'])

        # Two fully connected layers
        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_rcnn_seq:
            tensor = layer(tensor)

        tensor_output = layers_sensor[0](tensor)

        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = keras.Model(embed_model.inputs, tensor_output)

# Run My_Double_CNN with RoBERTa
tf_board = TensorBoard(log_dir='tf_dir/double_cnn_model',
                       histogram_freq=5,
                       update_freq='batch')
K.clear_session()
model = My_Double_CNN(bert_embedding)
model.fit(Xtr, ytr, Xva, yva,callbacks=[tf_board],epochs=30,batch_size=32)
report = model.evaluate(Xts, yts)

%load_ext tensorboard
%tensorboard --logdir tf_dir/double_cnn_model

# In SMP2018ECDTCorpus
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.tasks.classification import CNN_Model

train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')

tf_board = TensorBoard(log_dir='tf_dir/SMP2018ECDTCorpus_cnn_model',
                       histogram_freq=5,
                       update_freq='batch')
K.clear_session()
model = CNN_Model(bert_embedding)
model.fit(train_x, train_y, valid_x, valid_y,callbacks=[tf_board],epochs=30,batch_size=32)
report = model.evaluate(test_x, test_y)
print(report)

%load_ext tensorboard
%tensorboard --logdir tf_dir/SMP2018ECDTCorpus_cnn_model

# In SMP2018ECDTCorpus
train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')

tf_board = TensorBoard(log_dir='tf_dir/SMP2018ECDTCorpus_CNN_BiLSTM_model',
                       histogram_freq=5,
                       update_freq='batch')
K.clear_session()
# CNN + BiLSTM with RoBERTa
embed = BertEmbedding(bert_path)
model = MY_CNN_BILSTM_Model(bert_embedding)
model.fit(train_x, train_y, valid_x, valid_y,callbacks=[tf_board],epochs=30,batch_size=32)
report = model.evaluate(test_x, test_y)
print(report)

%load_ext tensorboard
%tensorboard --logdir tf_dir/SMP2018ECDTCorpus_CNN_BiLSTM_model
