from chapter01 import get_string_from_file
from chapter02 import get_words_from_file, configure_fonts_for_japanese
from chapter03 import get_words, bows_to_cfs, load_aozora_corpus, get_bows,  add_to_corpus,    get_weights, translate_bows, get_tfidfmodel_and_weights
from chapter04 import vsm_search, get_list_from_file
from chapter05 import top_n, get_pr_curve, get_average_precision


# Listing 9.2 #

from gensim.models import LsiModel


# Listing 9.4 #

from gensim.similarities import MatrixSimilarity

def lsi_search(texts, query, num_topics):
    # tfidfに基づいて語の重みを計算
    tfidf_model, dic, text_tfidf_weights = get_tfidfmodel_and_weights(texts)

    # LSIモデルを生成し，トピックの重みを計算
    lsi_model = LsiModel(corpus=text_tfidf_weights, id2word=dic,
                         num_topics=num_topics)
    lsi_weights = lsi_model[text_tfidf_weights]
    index = MatrixSimilarity(lsi_weights, num_features=len(dic))

    # queryのbag-of-wordsを作成し，重みを計算
    query_bows = get_bows([query], dic)
    query_tfidf_weights = get_weights(query_bows, dic, tfidf_model)
    query_lsi_weights = lsi_model[query_tfidf_weights]

    # 類似度計算
    sims = index[query_lsi_weights[0]]

    # 類似度で降順にソート
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)


# Listing 9.8 #

from gensim.models.nmf import Nmf

def nmf_search(texts, query, num_topics, passes=20, random_state=None):
    tfidf_model, dic, text_tfidf_weights = get_tfidfmodel_and_weights(texts)

    # NMFモデルを作成
    nmf_model = Nmf(corpus=text_tfidf_weights, id2word=dic, 
                    num_topics=num_topics, passes=passes, random_state=random_state)

    # TF・IDFによる文書ベクトルをトピックベースのベクトルに変換
    nmf_weights = nmf_model[text_tfidf_weights]

    index = MatrixSimilarity(nmf_weights, num_features=len(dic))

    # クエリのトピックベースのベクトルを作成
    query_bows = get_bows([query], dic)
    query_tfidf_weights = get_weights(query_bows, dic, tfidf_model)
    query_nmf_weights = nmf_model[query_tfidf_weights]

    # クエリとの類似性で文書をランキング
    sims = index[query_nmf_weights[0]]
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)


# Listing 9.11 #

from gensim.models import LdaModel


# Listing 9.14 #

def lda_search(texts, query, num_topics, passes=20, random_state=None):
    tfidf_model, dic, text_tfidf_weights = get_tfidfmodel_and_weights(texts)

    # LDAモデルを作成
    lda_model = LdaModel(corpus=text_tfidf_weights, id2word=dic,
                 num_topics=num_topics, passes=passes, random_state=random_state)

    lda_weights = lda_model[text_tfidf_weights]
    index = MatrixSimilarity(lda_weights, num_features=len(dic))

    query_bows = get_bows([query], dic)
    query_tfidf_weights = get_weights(query_bows, dic, tfidf_model)
    query_lda_weights = lda_model[query_tfidf_weights]

    sims = index[query_lda_weights[0]]
    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)






