# 关键词抽取

本实验用了两类共**5**种方法，分别是基于无监督与基于有监督（即机器学习）的两类方法，无监督用了TF-IDF和LDA，机器学习用的是RNN、LSTM和LR-SGD分类器，目前只在中文上运行成功，因为数据集中存在英文所以理论上英文也行。

## 运行环境

运行系统为Window10，运行软件为Pycharm，编程语言为python，python版本为3.8.3，numpy版本为1.18.5，jieba版本为0.42.1，gensim版本为3.8.3，scikit-learn版本为0.23.1，joblib版本为0.16.0，torch版本为1.7.0。

## 文件说明

`zh_wiki.py ` 和 `langconv.py`	用于繁体字转简体

`RNN & LSTM.py`	RNN和LSTM

`SGD.py`	SGD

`TF-IDF & LDA`	TF-IDF和LDA

`data文件夹`	用于存储数据集、停用词表和数据集特征提取后的nump矩阵

`result文件夹`	用于存储结果

`saved_model文件夹`	用于存储训练模型

`stopword.txt`	停用词表

`train_src.txt`、`train_trg.txt`、`valid_src.txt`、`valid_trg.txt`、`test_src.txt`、`test_trg.txt` 文本数据集

`dataset.npz`	数据集特征提取后的numpy矩阵保存文件

`lstm_best_0.pkl`	LSTM候选词个数不限时的最佳模型	`lstm_best_0.txt`	对应的输出结果

`lstm_best.pkl`	LSTM候选词个数为5时的最佳模型	`lstm_best.txt`	对应的输出结果

`rnn_best_0.pkl`	RNN候选词个数不限时的最佳模型	`rnn_best_0.txt`	对应的输出结果

`rnn_best.pkl`	RNN候选词个数为5时的最佳模型	`rnn_best.txt`	对应的输出结果

`sgd.pkl`	SGD模型（不论候选词个数为5还是不限都用这个模型）	`sgd_best.txt`、 `sgd_best_0.txt`	为对应输出的结果

`lad_best.txt`	LDA最佳结果

`tfidf_best.txt`	TF-IDF最佳结果

## 运行说明

### RNN

> RNN是一种特殊的神经网络结构，它之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。因此非常适合自然语言处理，于是本文用RNN来实现关键词抽取。

打开`RNN & LSTM.py`文件，将327行`model = torch.load('./saved_model/lstm_best.pkl')`改为`model = torch.load('./saved_model/rnn_best.pkl')`后运行即可。

若要训练，只需将319-324行的注释取消（代码如下），并将326-335行注释（代码如下），然后运行即可。

```python
train(model=rnn, #模型RNN就填rnn，LSTM就填lstm
      modelsavepath='./saved_model/rnn.pkl', #模型保存路径
      resultsavepath='./result/rnn.txt', #结果保存路径
      max=7.75, #当结果即mAP值大于max时保存结果
      num_epochs=1000 #运行总epoch次数
      )
```

```python
model = torch.load('./saved_model/lstm_best.pkl')
test(model=model.eval(), #传入训练好的模型
     test_word=npzfile['arr_2'], # 测试集词列表，主要用于结果输出
     test_X=test_X, # 测试集特征
     test_Y=test_Y, # 测试集标签
     doc=doc_list_test_src, # 测试集词表
     keywordslist=keyword_list(), # 关键词词表
     savepath='./result/lstm.txt', # 结果存储路径
     max=0) #当运行结果也就是mAP值大于max时，保存结果
```

### LSTM

> 长短期记忆网络是为了解决一般的RNN存在的权重指数级爆炸或消失的问题而专门设计出来的，LSTM的关键在于有三个门：（1）遗忘门。这个门主要是对上一个节点传进来的输入进行选择性忘记，会“忘记不重要的，记住重要的”。（2）输入门。这个门将这个阶段的输入有选择性地进行“记忆”。（3）输出门。这个门将决定哪些将会被当成当前状态的输出。本文用这个方法来弥补RNN的一些不足。

运行方法参考上面RNN。

### LR-SGD(Logistic Regression-Statistic Gradient Descent)

> Logistic Regression与回归联系并不大，反而多次出现在分类问题中，是常用于二分类的分类模型。它的实质含义是：事先假设数据集中的数据服从逻辑分布，然后运用极大似然法对参数进行估计。随机梯度下降算法在迭代过程中随机选择一个或几个样本的梯度来替代总体梯度，以达到降低计算复杂度的目的。本方法把词频（TF）、文本频率与逆文档频率指数（TF-IDF）、该词处于文本的位置（POS）作为特征，每个词分为是关键词和不是关键词两类，放入随机梯度下降分类器中训练，根据预测概率判断第一关键词。

打开`SGD.py`文件，直接运行即可，参数设置相关代码如下：

```python
sgd = SGDClassifier(loss='log', random_state=66) #随机数66
SGD(classifier=sgd, # 分类器
    train_features=npzfile['arr_0'], # 训练集特征矩阵形式
    train_labels=npzfile['arr_1'], # 训练集标签
    test_word=npzfile['arr_2'], # 测试集词列表，主要用于结果输出
    test_features=npzfile['arr_3'], #测试集特征矩阵形式
    test_labels=npzfile['arr_4'], #测试集标签矩阵形式
    load=True, #是否加载训练好的模型，否的话先训练并保存然后加载模型测试
    save=True, #是否保存结果
    savepath='./result/sgd.txt', #SGD结果保存路径
    end=0) #运行至第几行，设置0时为全部
```

### TF-IDF

> TF-IDF是一种统计方法，用以评估一个字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。因此用这个指标来提取出每句话的关键词，是非常合情合理的。

打开`TF-IDF & LDA`文件，直接运行即可，参数设置相关代码如下：

```python
TFIDF(savepath = './result/tfidf.txt', # TF-IDF结果保存路径
      keyword_num=5, # 候选词个数
      end = 0,  # end代表运行到第几行，设置为0时表示全部
      save=True) # 是否保存结果
```

### LDA

> LDA（Latent Dirichlet Allocation）是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。因此计算词的分布和文档的分布的相似度，取相似度最高的几个词作为关键词。

打开`TF-IDF & LDA`文件，直接运行即可，参数设置相关代码如下：

```python
LDA(savepath = './result/lda.txt', # LDA结果保存路径
    keyword_num=5, # 候选词个数
    end = 0, # end代表运行到第几行，设置为0时表示全部
    save=True) # 是否保存结果
```

## 结果

**实验指标**

1.F1@1:针对top-1的结果计算F1 score

得到输出的关键词序列后,获取其中的第一个关键词和GT中的第一个关键词作对比,从而计算出f1 score

2.F1@3:针对top-3的结果计算F1 score

得到关键词序列中的前三关键词与GT中的关键词做对比(不关心这三个关键词的相互顺序)

3.MAP（mean average precision）平均精度均值(参考：https://www.jianshu.com/p/82be426f776e)

先求出每个候选词在当前文本的精确率（TP/(TP+FP)），然后将这一段文本的所有候选词精确率求和并除以候选词个数得到该文本的平均精确率（AP），最后将所有文本的平均精确率相加后除以文本数，就得到平均精度均值（MAP）。

TF-IDF和LDA候选词个数为5，其他方法的候选词个数即预处理后的词个数时的结果（根据mAP升序排列）：

| 方法/指标 | F1@1  | F1@3  | mAP    |
| --------- | ----- | ----- | ------ |
| LR-SGD    | 4.51% | 4.32% | 5.95%  |
| RNN       | 7.30% | 5.87% | 6.82%  |
| TF-IDF    | 4.77% | 3.39% | 6.91%  |
| LSTM      | 6.91% | 6.74% | 7.16%  |
| LDA       | 7.28% | 5.64% | 11.30% |

所有方法候选词个数皆为5时的结果（根据mAP升序排列）:

| 方法/指标 | F1@1  | F1@3  | mAP    |
| --------- | ----- | ----- | ------ |
| TF-IDF    | 4.77% | 3.39% | 6.91%  |
| LR-SGD    | 4.54% | 4.36% | 7.33%  |
| RNN       | 6.95% | 5.62% | 8.88%  |
| LSTM      | 6.59% | 6.09% | 9.13%  |
| LDA       | 7.28% | 5.64% | 11.30% |

## 总结

根据实验结果发现：TF-IDF的F1@1指标略高于LR-SGD，F1@3要低于LR-SGD，但mAP高于LR-SGD，不过最好的还是LDA，而且LDA能有这样的成绩也并不意外，因为从其原理来看，是非常契合关键词抽取这一任务的，只是mAP相较于前两个指标有些虚高。

虽然一般情况下有监督的关键词抽取是要好于无监督的，但也看情况，这里分析原因主要有以下三点：1.TF-IDF和LDA的候选词个数对于前两个指标没有任何影响，因为前两个指标只考虑前三个预测关键词，所以将候选词个数调低对于mAP的提升较为明显，从而会导致TF-IDF和LDA前两个指标不是很高但mAP较高的情况，而有监督的方法也就是LR-SGD、RNN和LSTM，默认预处理后的词皆为候选词，故mAP值偏低，通过第二张表可以验证分析正确。2.训练集数量有限，且存在关键词并非出自原句，人名、剧名分开等问题3.特征提取不够，用词频、文本频率与逆文档频率指数（TF-IDF）、该词处于文本的位置作为特征，其实和TF-IDF方法没有太大的差别，也就多了位置属性，没有本质上的区别。还需改进的地方：1.增加命名实体识别和词性标注等特征2.多尝试几种机器学习分类器，找到最佳的那个3.修复训练集中存在的问题

总的来说，在时间有限或训练集有限的情况下，无监督的方法不失为一种好方法。但要从准确率的角度来看，首选有监督的方法，尤其是深度学习中的循环神经网络，其中长短时记忆网络效果最佳，因为它能考虑到词语的前后关系，这对于关键词抽取非常重要。

## 如何用自己的数据集训练

因为数据集不具有普适性，而本实验也是针对该数据集编写的，所以要用自己的数据集训练需要一些预处理：

最简单的办法就是把你的数据集弄成和我一样的，`train_src.txt`文件中存储分词后的文本以空格分隔，`train_trg.txt`文件中存储`train_src.txt`文件对应行的关键词同样以空格分隔，即使关键词不在原句也可以，格式如下所示：

`train_src.txt`文件中的第一行

```
taylorswift 在 格莱美 现场 演绎 了 alltoowell 真正 的 女神 穿 得 了 一身 华丽 的 礼服 弹得 了 一手 好 钢琴 还 甩 得 了 一头 好 秀发 根本 停不下来 优酷 音乐 红毯 直播
```

`train_trg.txt`文件中的第一行

```
56 届 格莱美
```

`test_src.txt`和`test_trg.txt`文件同样如此，`valid_src.txt`和`valid_trg.txt`文件本实验没有用到，所以省略。

将上述4个文件准备好后，可以选择覆盖原路径，这样运行的时候不用修改这一部分的路径，若要修改路径，在如下代码处修改：

```python
# 预处理加载好训练集和关键词
    doc_list_train_src = load_data(corpus_path='./data/train_src.txt')
    doc_list_test_src = load_data(corpus_path='./data/test_src.txt')
    keywords_list = keyword_list('./data/test_trg.txt') #它和doc_list_test_trg区别就是没有去停用词
```

TF-IDF和LDA方法在做了以上准备后运行`TF-IDF & LDA.py`文件即可，SGD、RNN和LSTM则需现将特征提取并保存为numpy矩阵后才能运行，所以先取消`SGD.py`文件283-291行的代码（如下所示）：

```python
train_features, train_labels, train_word = get_feature_label(end=0) # 训练集特征提取
# 测试集特征提取
test_features, test_labels, test_word = get_feature_label(readpath='./data/test_src.txt',
                                                           keypath='./data/test_trg.txt',
                                                           train=False, end=0)
# 保存特征为numpy矩阵
print("Numpy start to save")
np.savez('./data/dataset_own.npz', train_features, train_labels, test_word,
         test_features, test_labels, allow_pickle=True, fix_imports=True) 
print("Numpy save successfully")
```

然后记得修改294行的文件路径（如下所示）：

```python
npzfile = np.load('./data/dataset.npz', allow_pickle=True, fix_imports=True) #加载numpy矩阵
```

还有就是305行的`load=True`改为`False`，然后运行即可。

RNN和LSTM方法在有了上面保存的numpy矩阵后，只需修改`RNN & LSTM.py`文件281行（如**上**所示）的路径然后运行即可。

如果你的数据集是一行一句话（或其他情况）且不想额外弄成我这样子的数据集，其实只需在我的预处理中加入分词即可，这就要求你理解我预处理部分的代码，并稍作修改。千万别害怕，其实难度并不大，**代码里也基本都有注释**，静下心来慢慢看会看懂的。

理论上上面的方法应该有效，但不能成功运行可能是因为有些小地方我忽略了，比如路径的修改等等，大家多变通思考一下，希望我的代码能够帮到你们，有问题欢迎询问！