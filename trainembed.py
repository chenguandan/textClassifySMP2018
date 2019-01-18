import gensim, logging, json

class FileSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname):
            yield line.split( )


def train_word2vec(corpus , windows = 10 , sg = 1, iter = 5 , out_file =None ):
    sentences = FileSentences(corpus)
    model = gensim.models.Word2Vec(sentences, min_count = 0, size = 300 ,window=windows, sg=sg , iter=iter,workers = 4)
    model.save( out_file )


def prepare_data_char(filename, embed_corpus):
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    print('start')
    with open(embed_corpus, 'w', encoding='utf-8') as fout:
        with open(filename, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                label = labels[0]
                text = sample['内容']
                if label in label_dict:
                    fout.write(' '.join(text)+'\n')
    print('prepare data done.')

def prepare_data_term(filename, embed_corpus):
    import jieba
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    print('start')
    with open(embed_corpus, 'w', encoding='utf-8') as fout:
        with open(filename, encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                label = labels[0]
                text = sample['内容']
                terms = jieba.cut(text)
                if label in label_dict:
                    fout.write(' '.join(terms)+'\n')
    print('prepare data done.')


def char_main():
    embed_file = 'data/char_embed_corpus.txt'
    prepare_data_char('training.txt', embed_file)
    train_word2vec(embed_file, out_file='data/char_embed.model')
    print('done.')

def term_main():
    embed_file = 'data/term_embed_corpus.txt'
    prepare_data_term('training.txt', embed_file)
    train_word2vec(embed_file, out_file='data/term_embed.model')
    print('done.')

def char_main_more():
    embed_file = 'data/char_embed_corpus.txt'
    for windows in [3, 5, 8]:
        for sg in [0,1]:
            out_file = 'data/char_embed_{}_{}.model'.format(windows, sg)
            train_word2vec(embed_file, windows=windows, sg=sg, out_file=out_file)
    print('done.')

def term_main_more():
    embed_file = 'data/term_embed_corpus.txt'
    for windows in [3, 5, 8]:
        for sg in [0,1]:
            out_file = 'data/term_embed_{}_{}.model'.format(windows, sg)
            train_word2vec(embed_file, windows=windows, sg=sg, out_file=out_file)
    print('done.')


def char_ft_main_more():
    import fasttext
    embed_file = 'data/char_embed_corpus.txt'
    for windows in [3, 5, 8]:
        out_file = 'data/char_ft_embed_{}_{}.model'.format(windows, 1)
        model = fasttext.skipgram(embed_file, out_file, ws=windows, lr=0.1, dim=300, silent=0)
        out_file = 'data/char_ft_embed_{}_{}.model'.format(windows, 0)
        model = fasttext.cbow(embed_file, out_file, ws=windows, lr=0.1, dim=300, silent=0)
    print('done.')

def term_ft_main_more():
    import fasttext
    embed_file = 'data/term_embed_corpus.txt'
    for windows in [3, 5, 8]:
        out_file = 'data/term_ft_embed_{}_{}.model'.format(windows, 1)
        model = fasttext.skipgram(embed_file, out_file, ws=windows, lr=0.1, dim=300, silent=0)
        out_file = 'data/term_ft_embed_{}_{}.model'.format(windows, 0)
        model = fasttext.cbow(embed_file, out_file, ws = windows, lr=0.1, dim=300, silent=0)
    print('done.')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # char_main( )
    # term_main( )
    # char_main_more( )
    # term_main_more( )
    char_ft_main_more()
    term_ft_main_more()

