# !/usr/bin/env python
# coding: utf-8

## 导入本章所需要的模块
import torch.optim as optim
import numpy as np
import time
import jieba.posseg as psg
import torch
from torch import nn
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from langconv import *
import functools

# 繁体转简体
def TraditionalToSimplified(content):
    line = Converter("zh-hans").convert(content)
    return line

# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = './data/stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list

# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        # sentence = sentence.strip()
        # seg_list = sentence.split()
        sentence = TraditionalToSimplified(sentence)
        seg_list = sentence.strip().replace(';',' ').split()
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词(这里小于2的条件我去掉了，大家可以根据需求增加)
        if not word in stopword_list:
            filter_list.append(word)

    return filter_list


# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, corpus_path='./data/train_src.txt'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    doc_list = []
    inputs = open(corpus_path, 'r', encoding='utf-8')
    for line in inputs:
        seg_list = seg_to_list(line, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)
    inputs.close()
    return doc_list

def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a < b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

def keyword_list(keypath='./data/test_trg.txt'):
    inputs = open(keypath, 'r', encoding='utf-8')
    keyword_list =[]
    for line in inputs:
        line = TraditionalToSimplified(line)
        keyword_list.append(line.strip().replace(';',' ').split())
    inputs.close()
    return keyword_list

def get_first_three_keywords(keywordslist, row=1):
    length = len(keywordslist[row - 1])
    first_word = keywordslist[row - 1][0]
    m = min(length, 3)
    first_three_keywords = set(keywordslist[row - 1][:m])
    return first_word, first_three_keywords

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        input_dim:输入数据的维度
        hidden_dim: LSTM神经元个数
        layer_dim: LSTM的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  ## LSTM神经元个数
        self.layer_dim = layer_dim  ## LSTM的层数
        # LSTM ＋ 全连接层
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        out = self.fc1(r_out[:, -1, :])
        return out


# 搭建RNN模型
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        input_dim:输入数据的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim  ## RNN神经元个数
        self.layer_dim = layer_dim  ## RNN的层数
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu')
        # 连接全连阶层
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x:[batch, time_step, input_dim]
        # out:[batch, time_step, output_size]
        # h_n:[layer_dim, batch, hidden_dim]
        out, h_n = self.rnn(x, None)  # None表示h0会使用全0进行初始化
        # 选取最后一个时间点的out输出
        out = self.fc1(out[:, -1, :])
        return out

#对RNN或LSTM测试
def test(model, test_word, test_X, test_Y,
         doc, keywordslist, savepath, max=0):
    line_length = len(doc)
    f1_1 = 0
    f1_3 = 0
    sum_ap = 0  # 所有平均精确度的和，也就是每句话的平均精度之和
    c = 0
    string = ''
    for m in range(line_length):
        length = len(test_word[m])
        if length == 0:
            continue
        test_x = test_X[c:c + length]
        test_y = test_Y[c:c + length]

        test_data = Data.TensorDataset(test_x, test_y)
        test_loader = Data.DataLoader(
            dataset=test_data,  ## 使用的数据集
            batch_size=length,  # 批处理样本大小
            # shuffle=True,  # 每次迭代前打乱数据
            # num_workers = 1, # 使用两个进程
        )
        output_np = np.array([])
        test_y_np = test_y.detach().numpy()
        for step, (b_x, b_y) in enumerate(test_loader):
            # input :[batch, time_step, input_dim]
            xdata = b_x.view(-1, 1, 3)
            output = model(xdata)
            output_np = output.detach().numpy()
            # print(a[1])
            pre_lab = torch.argmax(output, 1)

        l = []
        for i in range(min(5,length)):
            t = []
            t.append(test_word[m][i])
            t.append(output_np[i][1])
            t.append(test_y_np[i])
            l.append(t)
        c = c + length

        l.sort(key=functools.cmp_to_key(cmp), reverse=True)
        sum_p = 0  # 每句话的精确率之和
        sum = 0  # top-i的True positive值
        for i in range(len(l)):
            # 计算每句话的精度之和
            sum = sum + l[i][2]
            sum_p = sum_p + sum / (i + 1)
        sum_ap = sum_ap + sum_p / len(l)
        first_keyword_pred = l[0][0]
        first_three_keywords_pred = set()
        for k, v, p in l[:min(3, length)]:
            first_three_keywords_pred.add(k)
        first_keyword, first_three_keywords = get_first_three_keywords(keywordslist, m + 1)
        str1 = str(m + 1) + '\t' + first_keyword_pred \
               + '\t' + '\t' + '\t' + first_keyword
        str2 = str(m + 1) + '\t' + str(first_three_keywords_pred) \
               + '\t' + '\t' + str(first_three_keywords)
        string = string + str1 + '\n' + str2 + '\n'
        # print(str1)
        # print(str2)
        if first_keyword_pred == first_keyword:
            f1_1 = f1_1 + 1
        if len(first_three_keywords_pred & first_three_keywords) == len(first_three_keywords):
            f1_3 = f1_3 + 1

        if (m + 1) == 0:
            break
    mAP = sum_ap / line_length * 100
    print("F1@1 : " + str(f1_1 / line_length * 100) + "%")
    print("F1@3 : " + str(f1_3 / line_length * 100) + "%")
    print("mAP : " + str(mAP) + "%")
    if mAP > max:
        outputs = open(savepath, 'w', encoding='utf-8')
        outputs.write('行数' + '\t' + '预测值' + '\t' + '\t' + '\t' + '真实值' + '\n')
        outputs.write(string)
        outputs.write("F1@1 : " + str(f1_1 / line_length * 100) + "%" + '\n')
        outputs.write("F1@3 : " + str(f1_3 / line_length * 100) + "%" + '\n')
        outputs.write("mAP : " + str(mAP) + "%")
        outputs.close()
        print("result save successfully")
    return mAP

# 对RNN或LSTM模型进行训练
def train(model, modelsavepath, resultsavepath, max, num_epochs=100):
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #采用自适应矩估计优化器
    criterion = nn.CrossEntropyLoss()  # 采用交叉熵损失函数
    keywordslist = keyword_list()
    for epoch in range(num_epochs): #循环num_epochs次数epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        model.train()  ## 设置模型为训练模式
        corrects = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            # input :[batch, time_step, input_dim]
            xdata = b_x.view(-1, 1, 3)
            output = model(xdata)
            # pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss.item() * b_x.size(0)
        mAP = test(model=model.eval(), test_word=npzfile['arr_2'],
                   test_X=test_X, test_Y=test_Y,
                   doc=doc_list_test_src,
                   keywordslist=keywordslist,
                   savepath=resultsavepath,
                   max=max)
        if mAP > max:
            max = mAP
            torch.save(model, modelsavepath)
            print('model save successfully')

if __name__ == '__main__':
    ### 数据准备
    print(time.strftime('Strat to prepare:%H:%M:%S', time.localtime(time.time())))
    doc_list_test_src = load_data(corpus_path='data/test_src.txt')
    print("Numpy start to load")
    npzfile = np.load('./data/dataset.npz', allow_pickle=True, fix_imports=True) # 加载数据集矩阵
    print("Numpy load successfully")
    #数据标准化，利于训练收敛
    mm = MinMaxScaler(feature_range=(0, 1))
    X = mm.fit_transform(npzfile['arr_0'])
    Y = mm.transform(npzfile['arr_3'])

    train_X = torch.from_numpy(X.astype(np.float32))
    train_Y = torch.from_numpy(npzfile['arr_1'].astype(np.int64))
    test_X = torch.from_numpy(Y.astype(np.float32))
    test_Y = torch.from_numpy(npzfile['arr_4'].astype(np.int64))

    train_data = Data.TensorDataset(train_X, train_Y)

    ## 定义一个数据加载器，将训练数据集进行批量处理
    train_loader = Data.DataLoader(
        dataset=train_data,  ## 使用的数据集
        batch_size=73,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
        # num_workers = 1, # 使用两个进程
    )

    rnn = RNN(input_dim=3, # 特征维度
              hidden_dim=7, # RNN神经元个数
              layer_dim=1, # RNN的层数
              output_dim=2 # 隐藏层输出的维度(2类)
             )

    lstm = LSTM(input_dim=3,  # 特征维度
                hidden_dim=7,  # LSTM神经元个数
                layer_dim=1,  # LSTM的层数
                output_dim=2  # 隐藏层输出的维度(2类)
                )

    print(time.strftime('Preparations have been done:%H:%M:%S', time.localtime(time.time())))

    # torch.manual_seed(42)
    # 训练模型
    # train(model=rnn, #模型RNN就填rnn，LSTM就填lstm
    #       modelsavepath='./saved_model/rnn.pkl', #模型保存路径
    #       resultsavepath='./result/rnn.txt', #结果保存路径
    #       max=7.75, #当结果即mAP值大于max时保存结果
    #       num_epochs=1000 #运行总epoch次数
    #       )

    # 加载训练好的模型并测试
    model = torch.load('./saved_model/lstm_best.pkl')
    test(model=model.eval(), #传入训练好的模型
         test_word=npzfile['arr_2'], # 测试集词列表，主要用于结果输出
         test_X=test_X, # 测试集特征
         test_Y=test_Y, # 测试集标签
         doc=doc_list_test_src, # 测试集词表
         keywordslist=keyword_list(), # 关键词词表
         savepath='./result/lstm.txt', # 结果存储路径
         max=0) #当运行结果也就是mAP值大于max时，保存结果

    print(time.strftime('End time:%H:%M:%S', time.localtime(time.time())))