from data_utils import TrainConfigure
import data_utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora
import gensim
import numpy as np
import logging

def prepare_tn_data(filename = 'data/topic_train.txt'):
    tn_conf = TrainConfigure()
    data_dict = data_utils.pickle_load(tn_conf.char_file)
    xterm = data_utils.pickle_load(tn_conf.term_file)
    term_vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
    reverse_dict = dict()
    for k, v in term_vocab_dict.items():
        reverse_dict[v] = k
    N = xterm.shape[0]
    with open(filename, 'w') as fout:
        for i in range(N):
            xi = xterm[i]
            term_list = []
            for idx in xi:
                if idx != 0 and idx != 1 and idx in reverse_dict:
                    term_list.append( reverse_dict[idx] )
            fout.write(' '.join(term_list)+'\n' )
    print('prepare data done.')


def get_list(dirname):
    docs = list()
    for line in open(dirname):
        docs.append(line.split())
    return docs


def train_LDA(filename = 'data/topic_train.txt'):
    # stop_words = ['on', 't', 'co', 'rt', 'https', 'http', 'a', 'the', 'of', 's']
    stop_words = ["%", "％", "》","《","他们", "该","·","-","（","）","(",")",";","]",":","/",";",
                  ",","我们","我","他们","但" ,"人" ,"这" ,"会","可以","没有","它","—","就","着","他们"]
    dictionary = corpora.Dictionary(line.split() for line in open(filename, encoding='utf-8'))
    dictionary.filter_extremes(no_below=1000, no_above=0.3)
    dictionary.save('./data/lda.dict')
    corpus = [dictionary.doc2bow(line.split()) for line in open(filename, encoding='utf-8')]
    corpora.MmCorpus.serialize('./data/lda.mm', corpus)
    corpus = None
    mm = corpora.MmCorpus('./data/lda.mm')
    lda = gensim.models.LdaModel(corpus=mm, id2word=dictionary, num_topics=20, passes=5, iterations=50)  # workers = 4
    lda.save("data/LDA20.model")


def get_vec( out_file, mode = "tn"):
    index = 0
    vec_len = 20
    dictionary = corpora.Dictionary.load('./data/lda.dict')
    lda = gensim.models.LdaModel.load('data/LDA20.model')
    if mode=="train":
        print('train')
        tn_conf = TrainConfigure()
    elif mode=="val":
        print('val')
        tn_conf = data_utils.ValidConfigure()
    else:
        print("test")
        tn_conf = data_utils.TestConfigure()
    # data_dict = data_utils.pickle_load(tn_conf.char_file)
    xterm = data_utils.pickle_load(tn_conf.term_file)
    term_vocab_dict = data_utils.pickle_load(tn_conf.term_dict)
    reverse_dict = dict()
    for k, v in term_vocab_dict.items():
        reverse_dict[v] = k

    all_lda = []
    for xi in xterm:
        doc = []
        for idx in xi:
            if idx != 0 and idx != 1 and idx in reverse_dict:
                doc.append(reverse_dict[idx])
        doc_bow = dictionary.doc2bow(doc)
        lda_vec_tmp = lda[doc_bow]
        lda_vec = np.zeros(vec_len)
        for (index, p) in lda_vec_tmp:
            lda_vec[index] = p
        index += 1
        all_lda.append( lda_vec )
    data_utils.pickle_dump(np.array(all_lda), out_file)
    print('done.')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare_tn_data( )

    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_LDA( )
    else:
        get_vec('data/lda_vec.pkl', mode="train")
        get_vec('data/lda_vec_val.pkl', mode="val")
        get_vec('data/lda_vec_test.pkl', mode="test")