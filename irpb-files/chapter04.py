from chapter01 import get_string_from_file
from chapter02 import get_words_from_file
from chapter03 import get_words, bows_to_cfs, load_aozora_corpus, get_bows,    add_to_corpus, get_weights, translate_bows, get_tfidfmodel_and_weights


# Listing 4.1 #

def jaccard(X, Y):
    x = set(X)
    y = set(Y)
    a = len(x.intersection(y))
    b = len(x.union(y))
    if b == 0:
        return 0
    else:
        return a/b


# Listing 4.3 #

from gensim import corpora, models


#  Listing 4.4 #

from gensim.similarities import MatrixSimilarity

def vsm_search(texts, query):
    tfidf_model, dic, text_weights = get_tfidfmodel_and_weights(texts)

    index = MatrixSimilarity(text_weights,  num_features=len(dic))

    # queryのbag-of-wordsを作成し，重みを計算
    query_bows = get_bows([query], dic)
    query_weights = get_weights(query_bows, dic, tfidf_model)

    # 類似度計算
    sims = index[query_weights[0]]

    # 類似度で降順にソート
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

def get_list_from_file(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        return f.read().split()





