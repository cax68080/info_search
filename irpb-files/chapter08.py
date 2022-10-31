from chapter01 import get_string_from_file
from chapter02 import get_words_from_file, configure_fonts_for_japanese
from chapter03 import get_words, bows_to_cfs, load_aozora_corpus, get_bows,  add_to_corpus,    get_weights, translate_bows, get_tfidfmodel_and_weights
from chapter04 import vsm_search, get_list_from_file


# Listing 8.2 #

def add_weights(dic, vec, weight=1.0):
    for (id, val) in vec:
        if not id in dic:
            dic[id] = 0
        dic[id] += weight*val

def Rocchio(query_vec, R_plus_vecs, R_minus_vecs,
    alpha=1.0, beta=0.75, gamma=0.15):

    # query_vec = [(id1, val1), (id2, val2), ...] から
    # { id1 : alpha*val1, id2 : alpha*val2, ,...} を計算 (8.1式の第1項)
    q = { id : alpha*val for (id, val) in query_vec }

    # 適合文書の文書ベクトルをqに反映させる (8.1式の第2項)
    n = len(R_plus_vecs)
    if n > 0:
        w = beta/n
        # R_plus_vecsの要素にwをかけて加算
        for v in R_plus_vecs:
            add_weights(q, v, weight=w)

    # 不適合文書の文書ベクトルをqに反映させる (8.1式の第3項)
    n = len(R_minus_vecs)
    if n > 0:
        w = -gamma/n
        for v in R_minus_vecs:
            add_weights(q, v, weight=w)

    # 辞書型のデータをbag-of-wordsフォーマットに変換
    return list(q.items())


# Listing 8.3 #

from gensim.similarities import MatrixSimilarity

def vsm_search_with_feedback(texts, query, R_plus, R_minus):
    tfidf_model, dic, text_weights = get_tfidfmodel_and_weights(texts)

    index = MatrixSimilarity(text_weights,  num_features=len(dic))
    query_bows = get_bows([query], dic)
    query_weights = get_weights(query_bows, dic, tfidf_model)

    # 適合/不適合文書のベクトルのリストを作成
    R_plus_vecs = [text_weights[i] for i in R_plus]
    R_minus_vecs = [text_weights[i] for i in R_minus]

    # Rocchioのアルゴリズムでクエリのベクトルquery_weights[0]を修正
    feedback_query = Rocchio(query_weights[0], R_plus_vecs, R_minus_vecs)

    # 修正したクエリとの類似度を計算
    sims = index[feedback_query]

    return sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
