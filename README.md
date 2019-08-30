
# Chinese Text Classification

文本分类（Text Classification）是自然语言处理中的一个重要应用技术，根据文档的内容或主题，自动识别文档所属的预先定义的类别标签。<br>
本文语料来自搜狗新闻文本 [下载链接](https://pan.baidu.com/s/1SMfx0X0-b6F8L9J6T5Hg2Q)，密码:dh4x。<br>
预训练词向量模型来自[GitHub：Chinese Word Vectors 上百种预训练中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)，下载地址：[Sogou News 300d](https://pan.baidu.com/s/1tUghuTno5yOvOx4LXA9-wg)。


<!-- TOC -->

* Chinese Text Classification
    * Part 1: 基于scikit-learn机器学习的文本分类方法
        * 1. 语料预处理
        * 2. 划分训练集与测试集
        * 3. TF-IDF文本特征提取
        * 4. 构建分类器
            * 朴素贝叶斯分类器
            * 逻辑回归LR
            * 支持向量机SVM
        * 5. 模型评估与比较
    * Part 2: 基于神经网络模型的文本分类方法
        * 1. 读取语料
        * 2. 语料预处理
            * 将文本转化为词向量矩阵
            * 划分训练集与校验集
        * 3. 构建模型
            * model 1 自己训练词权重向量
            * model 2 加载预训练模型权重
            * model 3 使用CNN进行文本分类
        * 4. 模型评估与比较

<!-- /TOC -->

## Part 1: 基于scikit-learn机器学习的文本分类方法

### 1. 语料预处理

搜狗新闻文本标签，`C000008`标签对应的新闻类别，为了便于理解，定义映射词典。

```python
category_labels = {
    'C000008': '_08_Finance',
    'C000010': '_10_IT',
    'C000013': '_13_Health',
    'C000014': '_14_Sports',
    'C000016': '_16_Travel',
    'C000020': '_20_Education',
    'C000022': '_22_Recruit',
    'C000023': '_23_Culture',
    'C000024': '_24_Military'
}
```

### 2. 划分训练集和测试集

将文本进行分词预处理，输出：训练语料数据(`X_train_data`)、训练语料标签(`y_train`)、测试语料数据(`X_test_data`)、测试语料标签(`y_test`)。

```python
X_train_data, y_train, X_test_data, y_test = load_datasets()
```
    label: _08_Finance, len: 1500
    label: _10_IT, len: 1500
    label: _13_Health, len: 1500
    label: _14_Sports, len: 1500
    label: _16_Travel, len: 1500
    label: _20_Education, len: 1500
    label: _22_Recruit, len: 1500
    label: _23_Culture, len: 1500
    label: _24_Military, len: 1500
    train corpus len: 13500

    label: _08_Finance, len: 490
    label: _10_IT, len: 490
    label: _13_Health, len: 490
    label: _14_Sports, len: 490
    label: _16_Travel, len: 490
    label: _20_Education, len: 490
    label: _22_Recruit, len: 490
    label: _23_Culture, len: 490
    label: _24_Military, len: 490
    test corpus len: 4410

### 3. TF-IDF文本特征提取

TF-IDF是一种统计方法，用以评估一字词对于一个文件集或者一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。意味着一个词语在一篇文章中出现的次数越多，同时在所有文档中出现的次数越少，越能代表该文章。

```python
stopwords = open('dict/stop_words.txt', encoding='utf-8').read().split()

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_data)
words = tfidf_vectorizer.get_feature_names()
```

### 4. 构建分类器

#### 朴素贝叶斯分类器

得到了训练样本的文本特征，现在可以训练出一个分类器，以用来对新的新闻文本进行分类。scikit-learn中提供了多种分类器，其中MultinomialNB比较适合于文本分类。

```python
mnb_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

%time mnb_clf.fit(X_train_data, y_train)
```

                   precision    recall  f1-score   support

      _08_Finance       0.88      0.90      0.89       477
           _10_IT       0.81      0.87      0.84       461
       _13_Health       0.82      0.90      0.86       451
       _14_Sports       0.98      1.00      0.99       480
       _16_Travel       0.89      0.91      0.90       480
    _20_Education       0.84      0.87      0.85       472
      _22_Recruit       0.90      0.73      0.80       606
      _23_Culture       0.80      0.83      0.82       476
     _24_Military       0.94      0.91      0.92       507

         accuracy                           0.87      4410
        macro avg       0.87      0.88      0.87      4410
     weighted avg       0.88      0.87      0.87      4410


#### 逻辑回归LR

尝试一下其他的分类器，比如`Logistic Regression`，训练新的分类器：

```python
lr_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])

%time lr_clf.fit(X_train_data, y_train)
```
                   precision    recall  f1-score   support

      _08_Finance       0.86      0.94      0.90       451
           _10_IT       0.86      0.87      0.86       484
       _13_Health       0.91      0.90      0.90       496
       _14_Sports       0.98      0.99      0.99       485
       _16_Travel       0.91      0.92      0.91       484
    _20_Education       0.86      0.91      0.88       464
      _22_Recruit       0.88      0.87      0.88       495
      _23_Culture       0.87      0.77      0.82       552
     _24_Military       0.96      0.94      0.95       499

         accuracy                           0.90      4410
        macro avg       0.90      0.90      0.90      4410
     weighted avg       0.90      0.90      0.90      4410
     
#### 支持向量机SVM

在传统机器学习中，SVM是做文本分类最好的工具.

```python
svm_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2')),
])

%time svm_clf.fit(X_train_data, y_train)
```
                     precision    recall  f1-score   support

        _08_Finance       0.87      0.95      0.91       449
             _10_IT       0.86      0.89      0.88       474
         _13_Health       0.92      0.91      0.92       495
         _14_Sports       1.00      0.99      0.99       493
         _16_Travel       0.94      0.92      0.93       501
      _20_Education       0.88      0.92      0.90       472
        _22_Recruit       0.93      0.89      0.91       515
        _23_Culture       0.89      0.85      0.87       512
       _24_Military       0.96      0.95      0.95       499

           accuracy                           0.92      4410
          macro avg       0.92      0.92      0.92      4410
       weighted avg       0.92      0.92      0.92      4410
       
### 5. 模型评估与比较

对新的文本需要进行分类.

```python
news_lastest = ["8月27日晚间，北京首钢篮球俱乐部官方宣布，正式与美籍华裔球员林书豪签约，林书豪将以外援身份，代表北京首钢队参加CBA联赛。同一时间，林书豪也在微博宣布：北京，我来了。过去9年时间里，华人林书豪在NBA经历了跌宕起伏的篮球生涯。从哈佛小子，到首位进入NBA的美籍华裔球员，再到千万身家的“林疯狂……与此同时，林书豪在国内获得了远高于在NBA的关注。",
                "在25日举行的七国工业国集团（G7）峰会上，美国总统特朗普在谈及朝鲜问题时重申近期朝鲜试射武器未违反协定，并指美韩军演是浪费金钱。据路透社报道，特朗普25日在与日本首相安倍晋三举行会谈时，谈及朝鲜问题。特朗普称他对朝鲜多次试射感到不满，但称发射“不违反任何协定”。",
                "8月14日，清华大学交叉信息院正式迎来了85名九字班新生。根据其官网的名单公示：今年5月18日校园开放日宣布成立的人工智能学堂班（简称智班）已完成了首次选拔，共录取30人。共有55名新生入学姚班。"]
X_new_data = [preprocess(doc) for doc in news_lastest]

mnb_clf.predict(X_new_data)
array(['_14_Sports', '_24_Military', '_20_Education'], dtype='<U13')
lr_clf.predict(X_new_data)
array(['_14_Sports', '_24_Military', '_20_Education'], dtype='<U13')
svm_clf.predict(X_new_data)
array(['_14_Sports', '_24_Military', '_14_Sports'], dtype='<U13')
```
比较各分类器的分类准确性

```
x_clf = ['mnb_clf', 'lr_clf', 'svm_clf']
y_clf = [mnb_score, lr_score, svm_score]
plt.bar(x_clf, y_clf)
plt.ylim(0.85, 0.92)
```
![](img/model_acc.png)



## Part 2: 基于神经网络模型的文本分类方法

### 1. 读取语料

### 2. 语料预处理

将文本转化为词向量矩阵
```
Shape of data tensor: (17910, 1000)
Shape of label tensor: (17910, 9)
```
data
```
array([[   0,    0,    0, ..., 1081, 2111,  218],
       [   0,    0,    0, ..., 3502,  508, 4917],
       [   0,    0,    0, ...,  193, 1287, 2759],
       ...,
       [   0,    0,    0, ...,  129,   31,  413],
       [   0,    0,    0, ...,  395, 3132,   46],
       [   0,    0,    0, ..., 4245, 1488,   23]], dtype=int32)
 ```
label
|序号|标签|名称|分类编码|
|:------:|:------:|:------:|:------:|
|0|C000008|Finance|[1, 0, 0, 0, 0, 0, 0, 0, 0]|
|1|C000010|IT|[0, 1, 0, 0, 0, 0, 0, 0, 0]|
|2|C000013|Health|[0, 0, 1, 0, 0, 0, 0, 0, 0]|
|3|C000014|Sports|[0, 0, 0, 1, 0, 0, 0, 0, 0]|
|4|C000016|Travel|[0, 0, 0, 0, 1, 0, 0, 0, 0]|
|5|C000020|Education|[0, 0, 0, 0, 0, 1, 0, 0, 0]|
|6|C000022|Recruit|[0, 0, 0, 0, 0, 0, 1, 0, 0]|
|7|C000023|Culture|[0, 0, 0, 0, 0, 0, 0, 1, 0]|
|8|C000024|Military|[0, 0, 0, 0, 0, 0, 0, 0, 1]|

### 3. 使用Keras对语料进行处理

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

MAX_SEQUENCE_LEN = 1000  # 文档限制长度
MAX_WORDS_NUM = 20000  # 词典的个数
VAL_SPLIT_RATIO = 0.2 # 验证集的比例

tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(len(word_index)) # all token found
# print(word_index.get('新闻')) # get word index
dict_swaped = lambda _dict: {val:key for (key, val) in _dict.items()}
word_dict = dict_swaped(word_index) # swap key-value
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN)

labels_categorical = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_categorical.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_categorical = labels_categorical[indices]

# split data by ratio
val_samples_num = int(VAL_SPLIT_RATIO * data.shape[0])

x_train = data[:-val_samples_num]
y_train = labels_categorical[:-val_samples_num]
x_val = data[-val_samples_num:]
y_val = labels_categorical[-val_samples_num:]
```

代码中`word_index`表示发现的所有词，得到的文本序列取的是`word_index`中前面20000个词对应的索引，文本序列集合中的所有词的索引号都在20000之前：

```python
len(data[data>=20000])
0
```

我们可以通过生成的词索引序列和对应的索引词典查看原始文本和对应的标签：

```python
# convert from index to origianl doc
for w_index in data[0]:
    if w_index != 0:
        print(word_dict[w_index], end=' ')
```

    昆虫 大自然 歌手 昆虫 口腔 发出 昆虫 界 著名 腹部 一对 是从 发出 外面 一对 弹性 称作 声 肌 相连 发音 肌 收缩 振动 声音 空间 响亮 传到 ５ ０ ０ 米 求婚 听到 发音 部位 发音 声音 两 张开 蚊子 一对 边缘 支撑 两只 每秒 ２ ５ ０ ～ ６ ０ ０ 次 推动 空气 往返 运动 发出 微弱 声 来源 语文 

```python
category_labels[dict_swaped(labels_index)[argmax(labels_categorical[0])]]
```

    '_20_Education'


### 4. 定义词嵌入矩阵

下面创建一个词嵌入矩阵，用来作为上述文本集合词典（只取序号在前`MAX_WORDS_NUM`的词，对应了比较常见的词）的词嵌入矩阵，矩阵维度是`(MAX_WORDS_NUM, EMBEDDING_DIM)`。矩阵的每一行`i`代表词典`word_index`中第`i`个词的词向量。这个词嵌入矩阵是预训练词向量的一个子集。我们的新闻语料中很可能有的词不在预训练词向量中，这样的词在这个词向量矩阵中对应的向量元素都设为零。还记得上面用`pad_sequence`补充的`0`元素么，它对应在词嵌入矩阵的向量也都是零。在本例中，20000个词有92.35%在预训练词向量中。


```python
EMBEDDING_DIM = 300 # embedding dimension
embedding_matrix = np.zeros((MAX_WORDS_NUM+1, EMBEDDING_DIM)) # row 0 for 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < MAX_WORDS_NUM:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
```

```python
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / MAX_WORDS_NUM
```

    0.9235

#### Embedding Layer

嵌入层的输入数据`sequence`向量的整数是文本中词的编码，前面看到这个获取序列编码的步骤使用了Keras的`Tokenizer API`来实现，如果不使用预训练词向量模型，嵌入层是用随机权重进行初始化，在训练中将学习到训练集中的所有词的权重，也就是词向量。在定义`Embedding`层，需要至少3个输入数据：

- `input_dim`：文本词典的大小，本例中就是`MAX_WORDS_NUM + 1`；
- `output_dim`：词嵌入空间的维度，就是词向量的长度，本例中对应`EMBEDDING_DIM`；
- `input_length`：这是输入序列的长度，本例中对应`MAX_SEQUENCE_LEN`。

本文中还多了两个输入参数`weights=[embedding_matrix]`和`trainable=False`，前者设置该层的嵌入矩阵为上面我们定义好的词嵌入矩阵，即不适用随机初始化的权重，后者设置为本层参数不可训练，即不会随着后面模型的训练而更改。这里涉及了`Embedding`层的几种使用方式：

- 从头开始训练出一个词向量，保存之后可以用在其他的训练任务中；
- 嵌入层作为深度学习的第一个隐藏层，本身就是深度学习模型训练的一部分；
- 加载预训练词向量模型，这是一种迁移学习，本文就是这样的示例。

### 5. 构建模型

Keras支持两种类型的模型结构：

- Sequential类，顺序模型，这个仅用于层的线性堆叠，最常见的网络架构
- Functional API，函数式API，用于层组成的有向无环图，可以构建任意形式的架构

为了有个对比，我们先不加载预训练模型，让模型自己训练词权重向量。`Flatten`层用来将输入“压平”，即把多维的输入一维化，这是嵌入层的输出转入全连接层(`Dense`)的必需的过渡。


```python
from keras.models import Sequential
from keras.layers import Dense, Flatten

input_dim = x_train.shape[1]

model1 = Sequential()
model1.add(Embedding(input_dim=MAX_WORDS_NUM+1, 
                    output_dim=EMBEDDING_DIM, 
                    input_length=MAX_SEQUENCE_LEN))
model1.add(Flatten())
model1.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(len(labels_index), activation='softmax'))

model1.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history1 = model1.fit(x_train, 
                    y_train,
                    epochs=30,
                    batch_size=128,
                    validation_data=(x_val, y_val))
```

    Train on 14328 samples, validate on 3582 samples
    Epoch 1/30
    14328/14328 [==============================] - 59s 4ms/step - loss: 3.1273 - acc: 0.2057 - val_loss: 1.9355 - val_acc: 0.2510
    Epoch 2/30
    14328/14328 [==============================] - 56s 4ms/step - loss: 2.0853 - acc: 0.3349 - val_loss: 1.8037 - val_acc: 0.3473
    Epoch 3/30
    14328/14328 [==============================] - 56s 4ms/step - loss: 1.7210 - acc: 0.4135 - val_loss: 1.2498 - val_acc: 0.5731
    ......
    Epoch 29/30
    14328/14328 [==============================] - 56s 4ms/step - loss: 0.5843 - acc: 0.8566 - val_loss: 1.3564 - val_acc: 0.6516
    Epoch 30/30
    14328/14328 [==============================] - 56s 4ms/step - loss: 0.5864 - acc: 0.8575 - val_loss: 0.5970 - val_acc: 0.8501


每个Keras层都提供了获取或设置本层权重参数的方法：

- `layer.get_weights()`：返回层的权重（`numpy array`）
- `layer.set_weights(weights)`：从`numpy array`中将权重加载到该层中，要求`numpy array`的形状与`layer.get_weights()`的形状相同


```python
embedding_custom = model1.layers[0].get_weights()[0]
embedding_custom
```

    array([[ 0.39893672, -0.9062594 ,  0.35500282, ..., -0.73564297,
             0.50492775, -0.39815223],
           [ 0.10640696,  0.18888871,  0.05909824, ..., -0.1642032 ,
            -0.02778293, -0.15340094],
           [ 0.06566656, -0.04023357,  0.1276007 , ...,  0.04459211,
             0.08887506,  0.05389333],
           ...,
           [-0.12710813, -0.08472785, -0.2296919 , ...,  0.0468552 ,
             0.12868881,  0.18596107],
           [-0.03790742,  0.09758633,  0.02123675, ..., -0.08180046,
             0.10254312,  0.01284804],
           [-0.0100647 ,  0.01180602,  0.00446023, ...,  0.04730382,
            -0.03696882,  0.00119566]], dtype=float32)



`get_weights`方法得到的就是词嵌入矩阵，如果本例中取的词典足够大，这样的词嵌入矩阵就可以保存下来，作为其他任务的预训练模型使用。通过`get_config()`可以获取每一层的配置信息：


```python
model1.layers[0].get_config()
```

    {'activity_regularizer': None,
     'batch_input_shape': (None, 1000),
     'dtype': 'float32',
     'embeddings_constraint': None,
     'embeddings_initializer': {'class_name': 'RandomUniform',
      'config': {'maxval': 0.05, 'minval': -0.05, 'seed': None}},
     'embeddings_regularizer': None,
     'input_dim': 20001,
     'input_length': 1000,
     'mask_zero': False,
     'name': 'embedding_13',
     'output_dim': 300,
     'trainable': True}

可以将模型训练的结果打印出来

```python
plot_history(history1)
```

![acc_loss_model1](img/acc_loss_model1.png)

第一个模型训练时间花了大约30分钟训练完30个epoch，这是因为模型需要训练嵌入层的参数，下面第二个模型在第一个模型基础上加载词嵌入矩阵，并将词嵌入矩阵设为不可训练，看是否可以提高训练的效率。


```python
from keras.models import Sequential
from keras.layers import Dense, Flatten

input_dim = x_train.shape[1]

model2 = Sequential()
model2.add(Embedding(input_dim=MAX_WORDS_NUM+1, 
                    output_dim=EMBEDDING_DIM, 
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LEN,
                    trainable=False))
model2.add(Flatten())
model2.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(len(labels_index), activation='softmax'))

model2.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model2.fit(x_train, 
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))
```

    Train on 14328 samples, validate on 3582 samples
    Epoch 1/10
    14328/14328 [==============================] - 37s 3ms/step - loss: 1.3124 - acc: 0.6989 - val_loss: 0.7446 - val_acc: 0.8088
    Epoch 2/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.2831 - acc: 0.9243 - val_loss: 0.5712 - val_acc: 0.8551
    Epoch 3/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.1183 - acc: 0.9704 - val_loss: 0.6261 - val_acc: 0.8624
    Epoch 4/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0664 - acc: 0.9801 - val_loss: 0.6897 - val_acc: 0.8607
    Epoch 5/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0549 - acc: 0.9824 - val_loss: 0.7199 - val_acc: 0.8660
    Epoch 6/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0508 - acc: 0.9849 - val_loss: 0.7261 - val_acc: 0.8582
    Epoch 7/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0513 - acc: 0.9865 - val_loss: 0.8251 - val_acc: 0.8585
    Epoch 8/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0452 - acc: 0.9858 - val_loss: 0.7891 - val_acc: 0.8707
    Epoch 9/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0469 - acc: 0.9865 - val_loss: 0.8663 - val_acc: 0.8680
    Epoch 10/10
    14328/14328 [==============================] - 35s 2ms/step - loss: 0.0418 - acc: 0.9867 - val_loss: 0.9048 - val_acc: 0.8640



```python
plot_history(history2)
```

![acc_loss_model2](img/acc_loss_model2.png)


从第二个模型训练结果可以看到预训练模型的加载可以大幅提高模型训练的效率，模型的验证准确度也提升的比较快，但是同时发现在训练集上出现了过拟合的情况。

第三个模型的结构来自于Keras作者的博客[示例](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)，这是CNN用于文本分类的例子。

```python
from keras.layers import Dense, Input, Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import Model

embedding_layer = Embedding(input_dim=MAX_WORDS_NUM+1,
                            output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LEN,
                            trainable=False)


sequence_input = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model3 = Model(sequence_input, preds)
model3.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history3 = model3.fit(x_train, 
                    y_train,
                    epochs=6,
                    batch_size=128,
                    validation_data=(x_val, y_val))
```

    Train on 14328 samples, validate on 3582 samples
    Epoch 1/6
    14328/14328 [==============================] - 77s 5ms/step - loss: 0.9943 - acc: 0.6719 - val_loss: 0.5129 - val_acc: 0.8582
    Epoch 2/6
    14328/14328 [==============================] - 76s 5ms/step - loss: 0.4841 - acc: 0.8571 - val_loss: 0.3929 - val_acc: 0.8841
    Epoch 3/6
    14328/14328 [==============================] - 77s 5ms/step - loss: 0.3483 - acc: 0.8917 - val_loss: 0.4022 - val_acc: 0.8724
    Epoch 4/6
    14328/14328 [==============================] - 77s 5ms/step - loss: 0.2763 - acc: 0.9100 - val_loss: 0.3441 - val_acc: 0.8942
    Epoch 5/6
    14328/14328 [==============================] - 76s 5ms/step - loss: 0.2194 - acc: 0.9259 - val_loss: 0.3014 - val_acc: 0.9107
    Epoch 6/6
    14328/14328 [==============================] - 77s 5ms/step - loss: 0.1749 - acc: 0.9387 - val_loss: 0.3895 - val_acc: 0.8788



```python
plot_history(history3)
```


![acc_loss_model3_cnn](img/acc_loss_model3_cnn.png)


通过加入池化层`MaxPooling1D`，降低了过拟合的情况。验证集上的准备度超过了前两个模型，也超过了传统机器学习方法。


### 参考资料

- [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [Keras Embedding Layers API](https://keras.io/layers/embeddings/)
- [How to Use Word Embedding Layers for Deep Learning with Keras](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
- [Practical Text Classification With Python and Keras](https://realpython.com/python-keras-text-classification/)
- [Francois Chollet: Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
