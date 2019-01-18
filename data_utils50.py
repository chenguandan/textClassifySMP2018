import json
from collections import Counter
import numpy as np
import re
class TrainConfigure(object):
    def __init__(self):
        self.MAX_LEN =50
        self.has_label = True
        self.filename = 'training.txt'
        self.new_dict = False
        self.char_dict = 'char_dict.pkl'
        self.char_dict_plain = 'char.dict'
        self.char_file = 'data50.dict'
        self.term_dict = 'term_dict.pkl'
        self.term_dict_plain = 'term.dict'
        self.term_file = 'data_terms50.pkl'
        self.feat_file = 'data_feat.pkl'
        self.feat_norm = 'feat.norm'

class ValidConfigure(object):
    def __init__(self):
        self.MAX_LEN = 50
        self.has_label = False
        self.filename = 'validation.txt'
        self.new_dict = False
        self.char_dict = 'char_dict.pkl'
        # self.char_dict_plain = 'char.dict'
        self.char_file = 'data_val50.dict'
        self.term_dict = 'term_dict.pkl'
        # self.term_dict_plain = 'term.dict'
        self.term_file = 'data_terms_val50.pkl'
        self.feat_file = 'data_feat_val.pkl'
        self.feat_norm = 'feat.norm'
        self.out_file = 'result.csv'


_URLRE = re.compile('(www|WWW)\\.[0-9%a-zA-Z\\.]+\\.(com|cn|org)')
import pickle
def pickle_dump(obj, fn ):
    with open(fn,'wb') as f:
        pickle.dump(obj, f)

def pickle_load(fn):
    obj = None
    with open(fn,'rb') as f:
        obj = pickle.load(f)
    return obj

def test():
    filename = 'sample.txt'
    with open(filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            print(sample)
def stat():
    #统计平均、最大、最小长度
    #统计各个标签的数量
    counter = Counter()
    #'标签','id','内容'
    filename = 'training.txt'
    max_len = 0
    sum_len = 0
    num_len = 0
    min_len = 1000
    with open(filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            label = sample['标签']
            text = sample['内容']
            num_len +=1
            sum_len += len(text)
            if len(text) > max_len:
                max_len = len(text)
            if len(text) < min_len and len(text)!=0:
                min_len = len(text)
            counter.update([label])
    for k, v in counter.items():
        print(k, v)
    print()
    #max: 33876 avg 676.716420458814 min 27 num 146421
    print('max:', max_len, 'avg', sum_len/num_len, 'min',min_len,'num', num_len )


def convert_char(conf,  V=40000, MAX_LEN=50):
    labels = [ '人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    char_dict = dict()
    char_counter = Counter()
    ids = []
    x = []
    y = []
    #统计各个字符的词频，保留高频字符
    print('start count char')
    with open(conf.filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            if conf.has_label:
                label = sample['标签']
            else:
                label = labels[0]
            text = sample['内容']
            id = sample['id']
            if label in label_dict:
                y.append(label_dict[label])
                ids.append(id)
            char_counter.update(text)
    __PAD = 0
    __UNK = 1
    if conf.new_dict:
        print('construct char dict.')
        #转化为index
        chars = char_counter.most_common(V)
        for index, (c,_) in enumerate(chars):
            char_dict[c] = index+2
        pickle_dump( char_dict, conf.char_dict)
        with open(conf.char_dict_plain, 'w') as fout:
            for c, index in sorted(char_dict.items(), key = lambda x:x[1] ):
                fout.write('{}\t{}\n'.format(c, index))
    else:
        print('load char dict.')
        char_dict = pickle_load(conf.char_dict)
    print('start convert and pad')
    #padding
    with open(conf.filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            if conf.has_label:
                label = sample['标签']
            else:
                label = labels[0]
            text = sample['内容']
            if label in label_dict:
                xi = []
                for c in text:
                    if c in char_dict:
                        xi.append( char_dict[c] )
                    else:
                        xi.append( __UNK )
                if len(xi) >= MAX_LEN:
                    xi = xi[:MAX_LEN]
                else:
                    xi = xi+[__PAD]*(MAX_LEN-len(xi))
                x.append( xi )

    x = np.array( x )
    y = np.array( y )
    data_dict = {'id':ids, 'x': x, 'y': y }
    print(x.shape, y.shape)
    pickle_dump( data_dict, conf.char_file )


def convert_terms(conf, V=40000, MAX_LEN=300):
    import jieba
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    term_dict = dict()
    term_counter = Counter()
    x = []
    # 统计各个字符的词频，保留高频字符
    print('start count char')
    num_index = 0
    with open(conf.filename, encoding='utf-8') as fin:
        for line in fin:
            num_index +=1
            if num_index % 10000==0:
                print('progress: ', num_index)
            sample = json.loads(line.strip())
            if conf.has_label:
                label = sample['标签']
            else:
                label = labels[0]
            text = sample['内容']
            if label in label_dict:
                term_list = jieba.cut(text)
                term_counter.update(term_list)
    __PAD = 0
    __UNK = 1
    if conf.new_dict:
        print('construct dict.')
        # 转化为index
        terms = term_counter.most_common(V)
        for index, (term, _) in enumerate(terms):
            term_dict[term] = index + 2
        pickle_dump(term_dict, conf.term_dict)
        with open(conf.term_dict_plain, 'w') as fout:
            for term, index in sorted(term_dict.items(), key=lambda x: x[1]):
                fout.write('{}\t{}\n'.format(term, index))
    else:
        print('load dict.')
        term_dict = pickle_load(conf.term_dict)
    print('start convert and pad')
    # padding
    with open(conf.filename, encoding='utf-8') as fin:
        for line in fin:
            num_index +=1
            if num_index % 10000==0:
                print('progress: ', num_index)
            sample = json.loads(line.strip())
            if conf.has_label:
                label = sample['标签']
            else:
                label = labels[0]
            text = sample['内容']
            if label in label_dict:
                term_list = jieba.cut(text)
                xi = []
                for term in term_list:
                    if term in term_dict:
                        xi.append(term_dict[term])
                    else:
                        xi.append(__UNK)
                if len(xi) >= MAX_LEN:
                    xi = xi[:MAX_LEN]
                else:
                    xi = xi + [__PAD] * (MAX_LEN - len(xi))
                if num_index % 10000 == 0:
                    print(xi)
                x.append(xi)
    x = np.array(x)
    print(x.shape)
    pickle_dump(x, conf.term_file)

def convert_feature(conf):
    """
    提取特征：
    :param filename:
    :return:
    """
    labels = [ '人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    x = []
    print('start')
    with open(conf.filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            if conf.has_label:
                label = sample['标签']
            else:
                label = labels[0]
            text = sample['内容']
            if label in label_dict:
                xi = []
                num_char = 0
                num_num = 0
                num_other = 0
                num_url = len( _URLRE.findall(text) )
                for c in text:
                    if (c>='A' and c<='Z') or (c>='a' and c<='z'):
                        num_char += 1
                    elif (c>='0' and c<='9'):
                        num_num += 1
                    else:
                        num_other += 1
                len_text = len(text)+1.0
                xi = [len(text), num_char, num_num, num_other,
                      (num_char+1.0)/len_text, (num_num+1.0)/len_text, (num_other+1.0)/len_text,
                      num_url]
                x.append(xi)
    x = np.array( x )
    pickle_dump(x, conf.feat_file)


def load_all_text(conf):
    """
    提取特征：
    :param filename:
    :return:
    """
    labels = [ '人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    texts = []
    print('start load text')
    with open(conf.filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            if conf.has_label:
                label = sample['标签']
            else:
                label = labels[0]
            text = sample['内容']
            if label in label_dict:
                texts.append( text )
    print('load text done.')
    return texts

def convert_sent_char(filename = 'training.txt', MAX_SENT=20, MAX_LEN=100):
    labels = [ '人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    x = []
    y = []
    char_dict = pickle_load('char_dict.pkl')
    print('start convert and pad sent')
    __PAD = 0
    __UNK = 1
    max_len = 0
    sum_len = 0
    num_len = 0
    min_len = 1000

    max_sent = 0
    sum_sent = 0
    num_sent = 0
    min_sent = 1000
    #padding
    #TODO 这样分句会出现很长的句子
    sent_end_set = set('。？！\n\r\t')
    line_num = 0
    with open(filename, encoding='utf-8') as fin:
        for line in fin:
            line_num += 1
            sample = json.loads(line.strip())
            label = sample['标签']
            text = sample['内容']
            if label in label_dict:
                y.append(label_dict[label])
                xi = []
                #划分句子
                sents = []
                starti = 0
                for i, c in enumerate(text):
                    if c in sent_end_set:
                        sents.append(text[starti:i+1])
                        starti = i+1
                if starti != len(text):
                    sents.append(text[starti:])
                sum_sent += len(sents)
                num_sent += 1
                if len(sents)>max_sent:
                    max_sent = len(sents)
                if len(sents) < min_sent:
                    min_sent = len(sents)

                for sent in sents:
                    num_len += 1
                    sum_len += len(sent)
                    if len(sent)>max_len:
                        max_len = len(sent)
                    if len(sent)<min_len:
                        min_len = len(sent)
                    senti = []
                    for c in sent:
                        if c in char_dict:
                            senti.append( char_dict[c] )
                        else:
                            senti.append( __UNK )
                    if len(senti) >= MAX_LEN:
                        senti = senti[:MAX_LEN]
                    else:
                        senti = senti+[__PAD]*(MAX_LEN-len(senti))
                    if len(xi) < MAX_SENT:
                        xi.append(senti)
                while len(xi) < MAX_SENT:
                    xi.append([__PAD]*MAX_LEN)
                x.append(xi)
    x = np.array( x )
    y = np.array( y )
    """
    sent 0 12.059540639662343 1918
    len 1 56.11461005680241 19324
    """
    print('sent',min_sent, sum_sent/num_sent, max_sent)
    print('len', min_len, sum_len/num_len, max_len)
    data_dict = {'x':x, 'y':y }
    print(x.shape, y.shape)
    pickle_dump( data_dict, 'data_sent.dict' )

def numurl_test():
    num_url = len(_URLRE.findall('www.baidu.com以及www.ab.cn,www.ab.org'))
    print(num_url)
    num_url = len(_URLRE.findall('www. baidu. com以及www. ab. cn,www. ab. org'))
    print(num_url)

    text = '192,Adds，中国'
    num_char = 0
    num_num = 0
    num_other = 0
    num_url = len(_URLRE.findall(text))
    for c in text:
        if (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z'):
            num_char += 1
        elif (c >= '0' and c <= '9'):
            num_num += 1
        else:
            num_other += 1
    print(num_char, num_num, num_other)

def load_embedding( vocab_dict, glove_file,
                    embed_dim = 300, skip_head= True, dump_path='data/embed.pkl' ):
    nb_vocab = max( vocab_dict.values( ) )
    import os
    if os.path.exists( dump_path ):
        matrix = pickle.load( open( dump_path, 'rb' ) )
    else:
        embeddings_index = dict( )
        with open(glove_file, encoding='utf-8') as f:
            if skip_head:
                next(f)
            for line in f:
                try:
                    values = line.split( )
                    if len(values) > embed_dim+1:
                        word = ' '.join(values[:-embed_dim])
                        print(word)
                    else:
                        word = values[0]
                    coefs = np.asarray(values[-embed_dim:], dtype='float32')
                    # coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    print(line)
        matrix = np.zeros( (nb_vocab+4, embed_dim) )
        nb_in_glove = 0
        include_set = set()
        for word, vec in embeddings_index.items():
            if word in vocab_dict:
                id = vocab_dict[word]
                matrix[id] = vec
                nb_in_glove += 1
                include_set.add( word )
        print('number of words in glove embedding: {}/{}'.format(nb_in_glove, len(vocab_dict)))
        pickle.dump(matrix, open(dump_path, 'wb'))
    return matrix

def  load_our_embedding( vocab_dict, model_file='data/char_embed.model',
                    embed_dim = 300, dump_path='data/our_char_embed.pkl' ):
    import gensim
    nb_vocab = max( vocab_dict.values( ) )
    import os
    if os.path.exists( dump_path ):
        matrix = pickle.load( open( dump_path, 'rb' ) )
    else:
        model = gensim.models.Word2Vec.load(model_file)
        matrix = np.zeros( (nb_vocab+4, embed_dim) )
        nb_in_glove = 0
        include_set = set()
        for word, id in vocab_dict.items():
            if word in model:
                matrix[id] = model[word]
                nb_in_glove += 1
                include_set.add( word )
        print('number of words in glove embedding: {}/{}'.format(nb_in_glove, len(vocab_dict)))
        pickle.dump(matrix, open(dump_path, 'wb'))
    return matrix

if __name__ == '__main__':
    import sys
    conf = TrainConfigure()
    if len(sys.argv)>1 and sys.argv[1] == 'val':
        conf = ValidConfigure()
        print('valid')

    # stat( )
    convert_char(conf, MAX_LEN=conf.MAX_LEN)
    convert_terms( conf, MAX_LEN=conf.MAX_LEN)
    # convert_feature( conf )
    # convert_sent_char( )
    # numurl_test( )