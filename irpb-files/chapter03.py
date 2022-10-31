from chapter01 import get_string_from_file
from chapter02 import create_wordcloud, get_japanese_fonts

# Listing 3.1 #

from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSStopFilter

# Listing 3.2 #

from janome.tokenfilter import POSKeepFilter

# Listing 3.3 #

def get_words(string, keep_pos=None):
    filters = []
    if keep_pos is None:
        filters.append(POSStopFilter(['記号']))       # 記号を除外
    else:
        filters.append(POSKeepFilter(keep_pos))       # 指定品詞を抽出
    filters.append(ExtractAttributeFilter('surface'))
    a = Analyzer(token_filters=filters)               # 後処理を指定
    return list(a.analyze(string))


# Listing 3.6 #

from gensim import corpora


# Listing 3.9 #

def build_corpus(file_list, dic_file=None, corpus_file=None):
    docs = []
    for f in file_list:
        text = get_string_from_file(f)
        words = get_words(text, keep_pos=['名詞'])
        docs.append(words)
        # ファイル名を表示
        print(f)
    dic = corpora.Dictionary(docs)
    if not (dic_file is None):
        dic.save(dic_file)
    bows = [dic.doc2bow(d) for d in docs]
    if not (corpus_file is None):
        corpora.MmCorpus.serialize(corpus_file, bows)
    return dic, bows



# Listing 3.10 #

def bows_to_cfs(bows):
    cfs = dict()
    for b in bows:
        for id, f in b:
            if not id in cfs:
                cfs[id] = 0
            cfs[id] += int(f)
    return cfs

def load_dictionary_and_corpus(dic_file, corpus_file):
    dic = corpora.Dictionary.load(dic_file)
    bows = list(corpora.MmCorpus(corpus_file))
    if not hasattr(dic, 'cfs'):
        dic.cfs = bows_to_cfs(bows)
    return dic, bows


# Listing 3.11 #

from gensim import models


# Listing 3.12 #

def load_aozora_corpus():
    return load_dictionary_and_corpus('data/aozora/aozora.dic',
                                      'data/aozora/aozora.mm')

def get_bows(texts, dic, allow_update=False):
    bows = []
    for text in texts:
        words = get_words(text, keep_pos=['名詞'])
        bow = dic.doc2bow(words, allow_update=allow_update)
        bows.append(bow)
    return bows

import copy

def add_to_corpus(texts, dic, bows, replicate=False):
    if replicate:
        dic = copy.copy(dic)
        bows = copy.copy(bows)
    texts_bows = get_bows(texts, dic, allow_update=True)
    bows.extend(texts_bows)
    return dic, bows, texts_bows


# Listing 3.13 #

def get_weights(bows, dic, model, surface=False, N=1000):
    # TF・IDFを計算
    weights = model[bows]
    # TF・IDFの値を基準に降順にソート．最大でN個を抽出
    weights = [sorted(w,key=lambda x:x[1], reverse=True)[:N] for w in weights]
    if surface:
        return [[(dic[x[0]], x[1]) for x in w] for w in weights]
    else:
        return weights


# Listing 3.15 #

def translate_bows(bows, table):
    return [[tuple([table[j[0]], j[1]]) for j in i if j[0] in table] for i in bows]

def get_tfidfmodel_and_weights(texts, use_aozora=True, pos=['名詞']):
    if use_aozora:
        dic, bows = load_aozora_corpus()
    else:
        dic = corpora.Dictionary()
        bows = []
    
    text_docs = [get_words(text, keep_pos=pos) for text in texts]
    text_bows = [dic.doc2bow(d, allow_update=True) for d in text_docs]
    bows.extend(text_bows)
    
    # textsに現れる語のidとtoken(表層形)のリストを作成
    text_ids = list(set([text_bows[i][j][0] for i in range(len(text_bows)) for j in range(len(text_bows[i]))]))
    text_tokens = [dic[i] for i in text_ids]
    
    # text_bowsにない語を削除．
    dic.filter_tokens(good_ids=text_ids)
    # 削除前後のIDの対応づけ
    # Y = id2id[X] として古いid X から新しいid Y が得られるようになる
    id2id = dict()
    for i in range(len(text_ids)):
        id2id[text_ids[i]] = dic.token2id[text_tokens[i]]
    
    # 語のIDが振り直されたのにあわせてbowを変換
    bows = translate_bows(bows, id2id)
    text_bows = translate_bows(text_bows, id2id)
    
    # TF・IDFモデルを作成
    tfidf_model = models.TfidfModel(bows, normalize=True)
    # モデルに基づいて重みを計算
    text_weights = get_weights(text_bows, dic, tfidf_model)
    
    return tfidf_model, dic, text_weights



