import fasttext
import json
import numpy as np
"""
fastText参数：
https://pypi.org/project/fasttext/#bag-of-tricks-for-efficient-text-classification
https://github.com/facebookresearch/fastText
"""
def write_data( fname, txts, ys ):
    with open( fname, 'w', encoding='utf-8') as fout:
        for txt, y  in zip( txts, ys):
            #TODO tokenize first?
            txt = txt.replace('\n', ' ')
            txt = txt.replace('\r',' ')
            label = '__label__{}'.format(y)
            fout.write(label+' '+txt+'\n')


def convert_data( fileout_tn, fileout_val, fileout_ts, tok_char=True):
    import jieba
    train_file = 'training_new.txt'
    test_file = 'validation_new.txt'
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    def read_data(filename, has_label):
        x = []
        y = []
        # 统计各个字符的词频，保留高频字符
        with open(filename, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                if has_label:
                    label = sample['标签']
                else:
                    label = labels[0]
                text = sample['内容']
                if label in label_dict:
                    y.append(label_dict[label])
                    if tok_char:
                        x.append(' '.join(text))
                    else:
                        terms_list = jieba.cut(text)
                        x.append( ' '.join(terms_list ) )
        return x, y
    x_tn, y_tn = read_data( train_file, has_label=True)
    x_tn = np.array(x_tn)
    y_tn = np.array(y_tn)
    indices = np.arange(len(x_tn))
    np.random.shuffle(indices)
    x_tn, y_tn = x_tn[indices], y_tn[indices]
    n_tn = int(0.95*len(x_tn))
    x_tn, y_tn, x_val, y_val = x_tn[:n_tn], y_tn[:n_tn], x_tn[n_tn:], y_tn[n_tn:]
    x_ts, y_ts = read_data(test_file, has_label=False)
    write_data( fileout_tn, x_tn, y_tn )
    write_data( fileout_val, x_val, y_val)
    write_data(fileout_ts, x_ts, y_ts)

def convert_main():
    fileout_tn = 'data/fasttest_train.txt'
    fileout_val = 'data/fasttest_val.txt'
    fileout_ts = 'data/fasttest_ts.txt'
    print('start convert data.')
    convert_data(fileout_tn, fileout_val, fileout_ts)
    print('convert char data done.')

    fileout_tn = 'data/fasttest_term_train.txt'
    fileout_val = 'data/fasttest_term_val.txt'
    fileout_ts = 'data/fasttest_term_ts.txt'
    convert_data(fileout_tn, fileout_val, fileout_ts, tok_char=False)
    print('convert term data done.')

def char_main():
    fileout_tn = 'data/fasttest_train.txt'
    fileout_val = 'data/fasttest_val.txt'
    fileout_ts = 'data/fasttest_ts.txt'
    # convert_data(fileout_tn, fileout_val, fileout_ts)
    # print('convert data done.')
    classifier = fasttext.supervised(fileout_tn, 'fasttextmodel', epoch=50,
                                     min_count= 10, word_ngrams=4, minn=0, maxn=0,
                                     dim =300, ws=5,
                                     bucket= 2000000)
    """
    0.9817
    epoch=25,min_count= 10, word_ngrams=4, minn=0, maxn=0,
                                     dim =500, ws=5,
                                     """
    result = classifier.test(fileout_val)
    # print('acc:', result.accuracy)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)
    # x_ts = []
    # with open(fileout_ts) as fin:
    #     for line in fin:
    #         idx = line.find(' ') + 1
    #         if idx is None or idx < 0:
    #             idx = 0
    #         x_ts.append(line[idx:])
    # y = classifier.predict(x_ts)
    # print(len(x_ts))
    # ya = []
    # for yi in y:
    #     ya.append(int(yi[0]))
    # print(len(y))

def term_main():
    fileout_tn = 'data/fasttest_term_train.txt'
    fileout_val = 'data/fasttest_term_val.txt'
    fileout_ts = 'data/fasttest_term_ts.txt'
    # convert_data(fileout_tn, fileout_val, fileout_ts, tok_char=False)
    # print('convert data done.')
    classifier = fasttext.supervised(fileout_tn, 'term_fasttextmodel',
                                     epoch=50,
                                     min_count= 10, word_ngrams=4, minn=0, maxn=0,
                                     dim =300, ws=5,
                                     bucket= 2000000)
    result = classifier.test(fileout_val)
    # print('acc:', result.accuracy)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)


def decode_y(y):
    """
    将fasttext的输出结果转化为numpy array
    :param y:
    :return:
    """
    yp = []
    for yi in y:
        ypi = [0.0, 0.0, 0.0, 0.0]
        for label, p in yi:
            if '__label__' in label:
                label = label[len('__label__'):]
            ypi[int(label)] = p
        yp.append(ypi)
    yp = np.array(yp)
    return yp



def predict_char():
    classifier = fasttext.load_model('fasttextmodel.bin')
    fileout_ts = 'data/fasttest_ts.txt'
    x_ts = []
    with open(fileout_ts) as fin:
        for line in fin:
            idx = line.find(' ') + 1
            if idx is None or idx < 0:
                idx = 0
            x_ts.append(line[idx:])
    prob = classifier.predict_proba(x_ts, k=4)#[('0', 0.998047)]
    # print(prob)
    return decode_y( prob )

def predict_term():
    classifier = fasttext.load_model('term_fasttextmodel.bin')
    fileout_ts = 'data/fasttest_term_ts.txt'
    x_ts = []
    with open(fileout_ts) as fin:
        for line in fin:
            idx = line.find(' ') + 1
            if idx is None or idx < 0:
                idx = 0
            x_ts.append(line[idx:])
    # y = classifier.predict(x_ts)
    # print(len(x_ts))
    # ya = []
    # for yi in y:
    #     ya.append(int(yi[0]))
    # print(len(y))
    prob = classifier.predict_proba(x_ts, k=4)#[('0', 0.998047)]

    return decode_y(prob)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        convert_main( )
    elif len(sys.argv) > 1 and sys.argv[1] == 'char':
        char_main( )
    else:
        term_main( )