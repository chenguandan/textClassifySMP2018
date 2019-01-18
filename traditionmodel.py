import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

def get_data(filename):
    Y = []
    X_sentences = []
    with open(filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            Y.append(sample['标签'])
            X_sentences.append(sample['内容'])

    print(len(X_sentences))
    classes_set = set(Y)
    print(classes_set)

    return X_sentences, Y
    #return X_sentences[0:1000], Y[0:1000]

def func_char(sentences, Y, gram_low, gram_high, n, vectorizer, classifer,scaler):
    print('# ' + str(n) + ' ' + vectorizer + ' ' + classifer + ' ' + scaler + ':')
    if vectorizer == 'Tfidf':
        vectorizer_char = TfidfVectorizer(encoding="utf8", analyzer='char', ngram_range=(gram_low, gram_high),
                                          max_features=n)
    elif vectorizer == 'Count':
        vectorizer_char = CountVectorizer(encoding="utf8", analyzer='char', ngram_range=(gram_low, gram_high),
                                          max_features=n)

    X = vectorizer_char.fit_transform(sentences)
    joblib.dump(vectorizer_char, 'Tradition_model-vectorizer-' + str(n) + '-' + vectorizer + '.pkl')

    features_char = vectorizer_char.get_feature_names()
    #print(features_char)
    print(len(features_char))

    if scaler=='MaxAbs':
        MaxAbs_scaler = preprocessing.MaxAbsScaler()
        X = MaxAbs_scaler.fit_transform(X)

    if classifer == 'Logistic':
        model = LogisticRegression(C=1e5)
    elif classifer == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif classifer == 'RandomForest':
        model = RandomForestClassifier(n_estimators=50, max_features=200, bootstrap=True)

    scores_word = cross_val_score(model, X, Y, cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0))
    print(scores_word)

    model.fit(X, Y)
    joblib.dump(model, 'Tradition_model-clf-' + str(n) + '-' + vectorizer + '-' + classifer + '-' + scaler + '.pkl')


def predict_char():
    # filename = 'C:\\Users\yanpe\Desktop\comp\\validation\\validation.txt'
    # vectorizer_char = joblib.load('Tradition_model-vectorizer-10000-Count.pkl')
    # model = joblib.load('Tradition_model-clf-10000-Count-Logistic.pkl')

    filename = 'validation.txt'
    vectorizer_char = joblib.load('Tradition_model-vectorizer-100000-Count.pkl')
    model = joblib.load('Tradition_model-clf-100000-Count-Logistic-none.pkl')

    X_sentences = []
    with open(filename, encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            X_sentences.append(sample['内容'])

    print(len(X_sentences))
    X = vectorizer_char.fit_transform(X_sentences)
    y = model.predict_proba(X)
    print(np.shape(y))

    return y

if __name__ == '__main__':
    #filename = 'C:\\Users\yanpe\Desktop\comp\\training\\training.txt'
    filename = 'training.txt'
    X_sentences, Y_labels = get_data(filename)
    #
    # func_char(X_sentences, Y_labels, 1, 8, 100000, 'Count', 'Logistic','none')
    func_char(X_sentences, Y_labels, 1, 6, 100000, 'Count', 'Logistic','none')
    # func_char(X_sentences, Y_labels, 1, 6, 50000, 'Count', 'Logistic','none')
    #predict_char()

