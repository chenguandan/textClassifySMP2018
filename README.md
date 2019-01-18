1.数据
训练数据：training.txt
验证数据：validataion.txt
格式为json
包括类别：'人类作者', '自动摘要', '机器作者', '机器翻译'

2.比赛网址
https://biendata.com/competition/smpeupt2018/leaderboard/
https://biendata.com/competition/smpeupt2018/final-leaderboard/

3.程序
data_utils.py   数据预处理（字典构建、分词、转化为index）
​    data_utils.py   val/test    表示测试集、验证集的数据转换
​    为训练相应的模型
train_topic_model.py    训练LDA主题模型，并且提取主题向量，存储在pickle文件中
hybridattmodel.py、hybridconvmodel.py等、conditionattmodel.py、conditiondensemamodel.py等
​    训练相应的模型，hybrid*.py融合了字符序列、单词序列、部分特征、LDA主题向量、位置embedding
​    condition*.py使用的条件模型，把y=f(x)转化为{0,1}=f(x,y_i)表示是否属于某个类别
​    使用方法示例为：hybridattmodel.py pe，这里的pe表示使用position embedding
conditiondualpathmode.py效果不好没有使用
charmodel.py、termmodel.py、attmodel.py、deepcnn.py只是用字符序列或者单词序列，效果不好，最终没有使用
trainembed.py   训练单词embedding
trainembed.py   分别训练不同类别的单词embedding
ensemble.py 读取不同的模型文件，集成预测结果，输出为result.csv
fasttextmodel.py    调用fasttext包进行分类


hybridmodel.py  效果不好，最终没有使用。采用了少量特征（在data_utils.py中提取的），包括：
	文本长度
	英文、数字、其他字符数量
	英文、数字、其他字符占总字符比例
	是否包含URL

hanmodel.py、Hpooling.py由于效果不好，没有使用。
