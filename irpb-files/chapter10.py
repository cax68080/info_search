from chapter01 import get_string_from_file
from chapter02 import get_words_from_file, configure_fonts_for_japanese
from chapter03 import get_words, bows_to_cfs, load_aozora_corpus, get_bows,  add_to_corpus,    get_weights, translate_bows, get_tfidfmodel_and_weights
from chapter04 import vsm_search, get_list_from_file
from chapter05 import top_n, get_pr_curve, get_average_precision


# Listing 10.1 #

from gensim.models import Word2Vec


# Listing 10.2 #

from gensim.models import KeyedVectors


# Listing 10.6 #

from gensim.models.doc2vec import TaggedDocument


# Listing 10.7 #

from gensim.models import Doc2Vec


# Listing 10.11 #

def d2v_search(model, texts, query):
    # textsの各要素を名詞の並びに変換
    docs = [get_words(text) if type(text) is str else text for text in texts]

    # queryも変換
    query_doc = get_words(query) if type(query) is str else query

    # docs[i]とquery_docをベクトル化して類似度を計算
    # そして，(i, 類似度) をi番目の要素とするリストを作成
    r = [(i, model.docvecs.similarity_unseen_docs(model, doc,  query_doc, steps=50))
             for i, doc in enumerate(docs)]
    # 類似度に関して降順にソートしたもの（ランキング）を返す
    return sorted(r, key=lambda x: x[1], reverse=True)

