from trainembed import *

def prepare_data_char_sep(filename, embed_corpus):
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    print('start')
    for labelindex, target_label in  enumerate(labels):
        print(target_label)
        with open(embed_corpus.format(labelindex), 'w', encoding='utf-8') as fout:
            with open(filename, encoding='utf-8') as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    label = sample['标签']
                    text = sample['内容']
                    if label == target_label:
                        fout.write(' '.join(text)+'\n')
    print('prepare data done.')

def prepare_data_term_sep(filename, embed_corpus):
    import jieba
    labels = ['人类作者', '自动摘要', '机器作者', '机器翻译']
    label_dict = dict([(label, i) for i, label in enumerate(labels)])
    print('start')
    for labelindex, target_label in enumerate(labels):
        print(target_label)
        with open(embed_corpus.format(labelindex), 'w', encoding='utf-8') as fout:
            with open(filename, encoding='utf-8') as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    label = sample['标签']
                    text = sample['内容']
                    if label == target_label:
                        terms = jieba.cut(text)
                        fout.write(' '.join(terms)+'\n')
    print('prepare data done.')

def char_main_more_sep():
    for labelindex in range(4):
        embed_file = 'data/char_embed_corpus{}.txt'.format(labelindex)
        # for windows in [3, 5, 8]:
        #     for sg in [0,1]:
        windows = 10
        sg = 1
        out_file = 'data/char_embed_{}_{}_{}.model'.format(labelindex, windows, sg)
        train_word2vec(embed_file, windows=windows, sg=sg, out_file=out_file)
    print('done.')

def term_main_more_sep():
    for labelindex in range(4):
        print('label: ',labelindex)
        embed_file = 'data/term_embed_corpus{}.txt'.format(labelindex)
        # for windows in [3, 5, 8]:
        #     for sg in [0,1]:
        windows = 10
        sg = 1
        out_file = 'data/term_embed_{}_{}_{}.model'.format(labelindex, windows, sg)
        train_word2vec(embed_file, windows=windows, sg=sg, out_file=out_file)
    print('done.')

if __name__ == '__main__':
    embed_file = 'data/term_embed_corpus{}.txt'
    prepare_data_term_sep('training_new.txt', embed_file)
    embed_file = 'data/char_embed_corpus{}.txt'
    prepare_data_char_sep('training_new.txt', embed_file)
    char_main_more_sep()
    term_main_more_sep()

