import datetime
import os
import re
from typing import Dict, Any

import kashgari
import keras
import tqdm
from kashgari.embeddings import BareEmbedding, BERTEmbedding, TransformerEmbedding
from kashgari.layers import L, AttentionWeightedAverageLayer
from kashgari.tasks.classification.base_model import BaseClassificationModel
from kashgari.tokenizer import BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow import keras

file_path = "./data/data.txt"

#预训练模型的访问路径
pretrained_model_path = "./pretrained_model/"
bert_model_path = {"bert": pretrained_model_path + "bert-base-chinese",
                   "albert": pretrained_model_path + "albert_base_zh_additional_36k_steps",
                   "albert_large": pretrained_model_path + "albert_large",
                   "albert_xxlarge": pretrained_model_path + "albert_xxlarge",
                   "baidu": pretrained_model_path + "baidu_ernie",
                   "xunfei": pretrained_model_path + "chinese_roberta_wwm_large_ext_L-24_H-1024_A-16",
                   "roberta": pretrained_model_path + "RoBERTa-large-clue"}

model_folder = './pretrained_model/albert_large'
checkpoint_path = os.path.join(model_folder, 'model.ckpt-best')
config_path = os.path.join(model_folder, 'albert_config.json')
vocab_path = os.path.join(model_folder, 'vocab_chinese.txt')

models = []

bert_version = 1
bert_name = "roberta"
bert_path = bert_model_path[bert_name]

exclude = False
exclude_label = ["A", "B"]

#一次训练所选取的样本数
batch_size = 32
#一个epoch是指把所有训练数据完整的过一遍
#epoch和epoch之间，训练数据会被shuffle
epochs = 5

# 有几个卷积核就会产生几个𝑓𝑒𝑎𝑡ℎ𝑒𝑟𝑚𝑎𝑝。输入数据经过卷积操作输出𝑓𝑒𝑎𝑡ℎ𝑒𝑟𝑚𝑎𝑝，从名字也可以知道（翻译就是特征图）这个是经过卷积核提取后得到的特征。
#
# 多个𝑓𝑒𝑎𝑡ℎ𝑒𝑟𝑚𝑎𝑝就意味着我们提取了多个特征值，这样或许就可以更加准确地识别数据。
class ClassificationModel(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.2
            },
            'bilstm_0': {
                'units': 64,
                'return_sequences': True
            },
            'conv_0': {
                # 卷积核又名过滤器(𝑓𝑖𝑙𝑡𝑒𝑟)。
                # 每个卷积核有三个属性：长宽深，这里一般深度不需要自己定义，深度是和输入的数据深度相同；
                'filters': 32,
                #卷积核的大小是5*5*1
                'kernel_size': 5,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
            },
            #开始的卷积核的值是随机的，之后每次的向后计算的过程中会得出这个图像的类别，当然这个第一次的结果大部分都是不准确的，之后经过𝑙𝑜𝑠𝑠𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛的作用，CNN会更新这些卷积核中的值，然后再来一次学习。这样经过多次的学习，CNN就会找到卷积核的最佳参数，使得提取的特征能准确区分这些图片，这样也就完成了CNN的学习过程。
        'maxpool': {},
            'attn': {},
            'average': {},
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
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_rcnn_seq = []
        # layers_rcnn_seq.append(L.SpatialDropout1D(**config['spatial_dropout']))
        layers_rcnn_seq.append(L.Bidirectional(L.LSTM(**config['bilstm_0'])))
        layers_rcnn_seq.append(L.Conv1D(**config['conv_0']))

        layers_sensor = []
        layers_sensor.append(L.GlobalMaxPooling1D())
        layers_sensor.append(AttentionWeightedAverageLayer())
        layers_sensor.append(L.GlobalAveragePooling1D())
        layer_concat = L.Concatenate(**config['concat'])

        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_rcnn_seq:
            tensor = layer(tensor)

        tensors_sensor = [layer(tensor) for layer in layers_sensor]
        tensor_output = layer_concat(tensors_sensor)
        # tensor_output = L.concatenate(tensor_sensors, **config['concat'])

        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = keras.Model(embed_model.inputs, tensor_output)


def load_data(filepath, bertpath, is_exclude, exclude_label_list):
    x_train, x_valid, x_test, y_train, y_valid, y_test = [], [], [], [], [], []
    content_list, label_list = [], []
    label_counter = {}

    global embed, tokenizer
    if bert_version == -1:
        embed = BareEmbedding(embedding_size=128, processor=kashgari.CLASSIFICATION, sequence_length=50)

    if bert_version == 1:
        embed = BERTEmbedding(bertpath, task=kashgari.CLASSIFICATION)
        tokenizer = embed.tokenizer
        embed.processor.add_bos_eos = False

    if bert_version == 2:
        tokenizer = BertTokenizer.load_from_vocab_file(vocab_path)
        embed = TransformerEmbedding(vocab_path, config_path, checkpoint_path,
                                     bert_type='albert', task=kashgari.CLASSIFICATION)

    pattern = '[，、。:：；/（）()《》“”"？,.;?·…0-9A-Za-z+=-]'
    lines = open(filepath, 'r', encoding='utf-8').read().splitlines()
    for line in tqdm.tqdm(lines):
        rows = line.split('\t')
        if len(rows) == 4:
            content = tokenizer.tokenize(re.sub(pattern, "", rows[0]))
            label = rows[1]
            if is_exclude:
                if label not in exclude_label_list:
                    x_train.append(content)
                    y_train.append(label)
                    if label not in label_list:
                        label_list.append(label)
                        label_counter[label] = 1
                    else:
                        label_counter[label] += 1
            else:
                #来到这里
                x_train.append(content)
                y_train.append(label)
                if label not in label_list:
                    label_list.append(label)
                    label_counter[label] = 1
                else:
                    label_counter[label] += 1

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    print("=== Data Summary ===")
    print("Train\t", len(x_train))
    print("Test\t", len(x_test))
    print("Label\t", len(label_list))
    for key, value in label_counter.items():
        print(key + '\t', value)
    return x_train, y_train, x_test, y_test, embed


if __name__ == '__main__':
    print("=== Data Loading ===")
    train_x, train_y, test_x, test_y, bert_embedding = load_data(file_path, bert_path, exclude, exclude_label)

    print("=== Model Loading ===")

    models.append(("CNN-BiLSTM-Attention", ClassificationModel(bert_embedding)))

    model_name = "saved_model"

    if len(models) != 0:
        print(len(models), "Model.")
        for model_ in models:
            print(model_[0])

        for model_ in models:
            print("=== Model Training ===")
            model_name = model_[0]
            model = model_[1]

            print("Pretrained Model:", bert_name)
            print("Classification Model:", model_name)
            print("Batch Size: {}, Epochs: {}".format(str(batch_size), str(epochs)))
            model.fit(train_x, train_y, test_x, test_y, batch_size, epochs)

            print("=== Model Evaluating ===")
            model.evaluate(test_x, test_y)
            report = model.evaluate(test_x, test_y, output_dict=True)
            # print("Precision:", precision_score(y_data, y_pred, average="weighted"))
            # print("Recall:", recall_score(y_data, y_pred, average="weighted"))
            # print("F1:", f1_score(y_data, y_pred, average="weighted"))

            print("=== Model Saving ===")
            datatime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_prefix = bert_name + "_" + model_name + "_" + str(batch_size) + "_" + str(epochs) + "_" + datatime
            pred_record_name = "./pred_record/" + file_prefix + ".txt"
            model.save('./saved_model/' + file_prefix)
            with open(pred_record_name, 'a+') as f:
                f.write(str(report))
            # with open(pred_record_name, 'a+') as f:
            #     for i in range(len(y_data)):
            #         f.write(y_data[i] + '\t' + y_pred[i] + '\n')
            print(file_prefix + " Done!")
            # plot_model(model, to_file='./saved_model/' + file_prefix + datatime + '/model.png')

    else:
        loaded_model = kashgari.utils.load_model('saved_model/' + model_name)
        print(test_x)
        print(test_y)
        print(loaded_model.predict_top_k_class(test_x))
