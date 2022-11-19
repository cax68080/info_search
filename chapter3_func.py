#!/usr/bin/env python
# coding: utf-8
# エンコーディングに応じて適切に読み込むget_string_from_file関数
import chardet
from gensim import corpora

def get_string_from_file(filename):
    with open(filename,'rb') as f:
        d = f.read()
        e = chardet.detect(d)['encoding']
        # 推定できなかったときはUTF-8を設定する
        if e == None:
            e = 'UTF-8'
        return d.decode(e)


# 日本語フォントの設定
import matplotlib.font_manager as fm

japanese_font_candidates = ['Hiragino Maru Gothic Pro','Yu Gothic','Arial Unicode MS','Meirio','Takao','IPAexGothic','IPAPGothic','VL PGothic','Noto Sans CJK JP']

def get_japanese_fonts(candidates=japanese_font_candidates):
    fonts = []
    for f in fm.findSystemFonts():
        p = fm.FontProperties(fname=f)
        try:
            n = p.get_name()
            if n in candidates:
                fonts.append(f)
        except RuntimeError:
            pass
    # サンプルデータアーカイブに含まれているIPAexフォントを追加
    fonts.append('./irpb-files/font/ipaexg.ttf')
    return fonts



# get_words関数
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSStopFilter
from janome.tokenfilter import POSKeepFilter

def get_words(string,keep_pos=None):
    filters = []
    if keep_pos is None:
        filters.append(POSStopFilter(['記号'])) #記号を除外
    else:
        filters.append(POSKeepFilter(keep_pos)) #指定品詞を抽出
    filters.append(ExtractAttributeFilter('surface'))
    a = Analyzer(token_filters=filters) #後処理を指定
    return list(a.analyze(string))


# ## 3.2 TF・IDF

# コーパスの構築を行うbuild_corpus関数
def build_corpus(file_list,dic_file=None,corpus_file=None):
    docs = []
    for f in file_list:
        text = get_string_from_file(f)
        words = get_words(text,keep_pos=['名詞'])
        docs.append(words)
        # ファイル名を表示
        print(f)
    dic = corpora.Dictionary(docs)
    if not (dic_file is None):
        dic.save(dic_file)
    bows = [dic.doc2bow(d) for d in docs]
    if not (corpus_file is None):
        corpora.MmCorpus.serialize(corpus_file,bows)
    return dic,bows

# 辞書とコーパスを読み込むload_dictionary_and_corpus関数
def bows_to_cfs(bows):
    cfs = dict()
    for b in bows:
        for id,f in b:
            if not id in cfs:
                cfs[id] = 0
            cfs[id] += int(f)
    return cfs

#def load_dictionary_and_corpus(dic_files,corpus_file):
#    from gensim import corpora
#    dic = corpora.Dictionary.load(dic_files)
#    bows = list(corpora.MmCorpus(corpus_file))
#    if not hasattr(dic,'cfs'):
#        dic.cfs = bows_to_cfs(bows)
#    return dic,bows

# TF・IDFの計算
from gensim import models

# コーパス作成のための関数
def load_aozora_corpus():
    return load_dictionary_and_corpus('./irpb-files/data/aozora/aozora.dic','./irpb-files/data/aozora/aozora.mm')

def get_bows(texts,dic,allow_update=False):
    bows = []
    for text in texts:
        words = get_words(text,keep_pos=['名詞'])
        bow = dic.doc2bow(words,allow_update=allow_update)
        bows.append(bow)
    return bows

import copy

def add_to_corpus(texts,dic,bows,replicate=False):
    if replicate:
        dic = copy.copy(dic)
        bows = copy.copy(bows)
    texts_bows = get_bows(texts,dic,allow_update=True)
    bows.extend(texts_bows)
    return dic,bows,texts_bows


def get_weights(bows,dic,tfidf_model,surface=False,N=1000):
    # TF・IDFを計算
    weights = tfidf_model[bows]
    # TF・IDFの値を基準に降順にソート、最大でN個を抽出
    weights = [sorted(w,key=lambda x:x[1],reverse=True)[:N] for w in weights]
    if surface:
        return [[(dic[x[0]],x[1]) for x in w] for w in weights]
    else:
        return weights


#get_tfidfmodel_and_weights関数
def translate_bows(bows,table):
    return [[tuple([table[j[0]],j[1]]) for j in i if j[0] in table] for i in bows]

def get_tfidfmodel_and_weights(texts,use_aozora=True,pos=['名詞']):
    from gensim import corpora
    if use_aozora:
        dic,bows = load_aozora_corpus()
    else:
        dic = corpora.Dictionary()
        bows = []
    text_docs = [get_words(text,keep_pos=pos) for text in texts]
    text_bows = [dic.doc2bow(d,allow_update=True) for d in text_docs]
    bows.extend(text_bows)
    
    # textsに現れる語のidとtoken(表層形)のリストを作成
    text_ids = list(set([text_bows[i][j][0] for i in range(len(text_bows)) for j in range(len(text_bows[i]))]))
    text_tokens = [dic[i] for i in text_ids]
    
    # text_bowsにない語を削除
    dic.filter_tokens(good_ids=text_ids)
    # 削除前後のID対応付け
    # Y = id2id[X]として古いid Xから新しいid Yが得られるようになる。
    id2id = dict()
    for i in range(len(text_ids)):
        id2id[text_ids[i]] = dic.token2id[text_tokens[i]]
        
    # 語のIDが振り直されたのにあわせてbowを変換
    bows = translate_bows(bows,id2id)
    text_bows = translate_bows(text_bows,id2id)
    
    # TF・IDFモデルを作成
    tfidf_model = models.TfidfModel(bows,normalize=True)
    # モデルに基づいて重みを計算
    text_weights = get_weights(text_bows,dic,tfidf_model)
    
    return tfidf_model,dic,text_weights
