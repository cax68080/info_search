# エンコーディングに応じて適切に読み込むget_string_from_file関数
import chardet

def get_string_from_file(filename):
    with open(filename,'rb') as f:
        d = f.read()
        e = chardet.detect(d)['encoding']
        # 推定できなかったときはUTF-8で
        if e == None:
            e = 'UTF-8'
        return d.decode(e)

from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSKeepFilter
from janome.tokenfilter import POSStopFilter

#def get_words(s):
#    a = Analyzer(token_filters=[POSStopFilter(['記号']),ExtractAttributeFilter('surface')])
#    return list(a.analyze(s))

def get_words_from_file(f):
    return get_words(get_string_from_file)

# 分かち書きに基づく検索を行うgat_m_snipet_from_file関数
from janome.tokenizer import Tokenizer

def get_m_snippet_from_file(filename,query,width=3):
    t = Tokenizer(wakati=True)
    qlist = list(t.tokenize((query)))
    qlen = len(qlist)
    s = c1.get_string_from_file(filename)
    slist = list(t.tokenize(s))
    for i in [k for k,v in enumerate(slist) if v == qlist[0]]:
        if qlist == slist[i:i + qlen]:
            return ''.join(slist[max(0,i - width):i + width + qlen])
    return None
# get_words関数(修正版)
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSStopFilter
from janome.tokenfilter import POSKeepFilter
from gensim import models
from gensim import corpora

def get_words(string,keep_pos=None):
    filters = []
    if keep_pos is None:
        filters.append(POSStopFilter(['記号']))
    else:
        filters.append(POSKeepFilter(keep_pos))
    filters.append(ExtractAttributeFilter('surface'))
    a = Analyzer(token_filters=filters)
    return list(a.analyze(string))

# コーパスの構築を行うbuild_corpus関数
def build_corpus(file_list,dic_file=None,corpus_file=None):
    docs = []
    for f in file_list:
        text = c1.get_string_from_file(f)
        words = get_words(text,keep_pos['名詞'])
        docs.append(words)
        # ファイル名を表示する
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

def load_dictionary_and_corpus(dic_file,corpus_file):
    dic = corpora.Dictionary.load(dic_file)
    bows = list(corpora.MmCorpus(corpus_file))
    if not hasattr(dic,'cfs'):
        dic.cfs = bows_to_cfs(bows)
    return dic,bows

# コーパス作成のための関数
def load_aozora_corpus():
    return load_dictionary_and_corpus('irpb-files/data/aozora/aozora.dic','irpb-files/data/aozora/aozora.mm')

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
    # TF・IDFを計算する
    weights = tfidf_model[bows]
    # TF・IDFの値を基準に降順にソート、最大でN個を抽出する
    weights = [sorted(w,key=lambda x:x[1],reverse=True)[:N] for w in weights]
    if surface:
        return [[(dic[x[0]],x[1]) for x in w] for w in weights]
    else:
        return weights

# TF・IDFの重みを計算するget_tfidfmodel_and_weights関数
def translate_bows(bows,table):
    return [[tuple([table[j[0]],j[1]]) for j in i if j[0] in table] for i in bows]

def get_tfidfmodel_and_weights(texts,use_aozora=True,pos=['名詞']):
    if use_aozora:
        dic,bows = load_aozora_corpus()
    else:
        dic = corpora.Dictionary()
        bows = []
        
    text_docs = [get_words(text,keep_pos=pos) for text in texts]
    text_bows = [dic.doc2bow(d,allow_update=True) for d in text_docs]
    bows.extend(text_bows)
    
    # textsに現れる語のidとtoken(表層形)のリストを作成する
    text_ids = list(set([text_bows[i][j][0] for i in range(len(text_bows)) for j in range(len(text_bows[i]))]))
    text_tokens = [dic[i] for i in text_ids]
    
    # text_bowsにない語を削除する
    dic.filter_tokens(good_ids=text_ids)
    # 削除前後のIDを対応付ける
    # Y = id2id[X]として古いid Xから新しいid Yが得られるようになる
    id2id = dict()
    for i in range(len(text_ids)):
        id2id[text_ids[i]] = dic.token2id[text_tokens[i]]
        
    # 語のIDが振りなおされたのに合わせてbowを変換する
    bows = translate_bows(bows,id2id)
    text_bows = translate_bows(text_bows,id2id)
    
    # TF・IDFモデルを作成する
    tfidf_model = models.TfidfModel(bows,normalize=True)
    # モデルに基づいて重みを計算する
    text_weights = get_weights(text_bows,dic,tfidf_model)
    
    return tfidf_model,dic,text_weights

# jaccard関数
# X,Yはiterative
def jaccard(X,Y):
    x = set(X)
    y = set(Y)
    a = len(x.intersection(y))
    b = len(x.union(y))
    if b == 0:
        return 0
    else:
        return a / b

# コサイン類似度を計算するvsm_search関数

from gensim.similarities import MatrixSimilarity

def vsm_search(texts,query):
    tfidf_model,dic,text_weights = get_tfidfmodel_and_weights(texts)
    
    index = MatrixSimilarity(text_weights,num_features=len(dic))
    
    # queryのbag-of-wordsを作成し、重みを計算
    query_bows = get_bows([query],dic)
    query_weights = get_weights(query_bows,dic,tfidf_model)
    
    # 類似度計算
    sims = index[query_weights[0]]
    
    # 類似度で降順にソートする
    return sorted(enumerate(sims),key=lambda x: x[1],reverse=True)

# ファイルに保存されているリストを読みだす
def get_list_from_file(file_name):
    with open(file_name,'r',encoding='UTF-8') as f:
        return f.read().split()
    

# 類似度が閾値より大きいものを適合文書と判断するselect_by_threshold関数
def select_by_threshold(r,threshold=0.0):
    # rの長さ分の０を要素とするリストを作成する
    answer = [0] * len(r)
    
    for i in r:
        # 類似度がthresholdより大きいときr[文書番号]を1にする
        if i[1] > threshold:
            answer[i[0]] = 1
    return answer

# 正解と検索結果より、評価である3つの数値を出力する関数
from sklearn.metrics import precision_score,recall_score,f1_score

def print_scores(right_answer,my_answer):
    print('precision %.4f' % precision_score(right_answer,my_answer))
    print('recall    %.4f' % recall_score(right_answer,my_answer))
    print('f-measure %.4f' % f1_score(right_answer,my_answer))
    
# ランキング上位を取り出す関数
def top_n(r,n):
    answer = [0] * len(r)
    for i in range(n):
        answer[r[i]] = 1
    return answer

# 適合率のリストと再現率のリストを作成する関数
def get_pr_curve(ranking,answer):
    # top_n(ranking,i)の適合率と再現率をそれぞれprecision[i]とrecall[i]に格納する
    # precision[0] = 1、recall[0] = 0とする
    precision = [1]
    recall = [0]
    for i in range(1,len(ranking) + 1):
        x = top_n(ranking,i)
        precision.append(precision_score(answer,x))
        recall.append(recall_score(answer,x))
    return precision,recall

#適合率－再現率曲線を描く関数
import matplotlib.pyplot as plt
%matplotlib inline

def draw_pr_curve(ranking,answer):
    precision,recall = get_pr_curve(ranking,answer)
    # グラフの描画範囲を設定
    plt.xlim(-0.05,1.05)
    plt.ylim(0.0,1.1)
    # 各軸のラベルを設定する。x軸に再現率、y軸に適合率
    plt.xlabel('recall')
    plt.ylabel('precision')
    # 曲線の下を塗りつぶす
    plt.fill_between(recall,precision,0,facecolor='#FFFFCC')
    # 曲線を点と線で描く
    plt.plot(recall,precision,'o-')
    plt.show()
    
# 平均適合率を取得する関数
def get_average_precision(ranking,answer):
    precision,recall = get_pr_curve(ranking,answer)
    ap = 0.0
    # (r[i - 1],0),(r[i - 1],p[i - 1]),(r[i],0),(r[i],p[i])で
    # 囲まれる面積をそれぞれapに加算
    for i in range(1,len(precision)):
        ap += (recall[i] - recall[i - 1]) * (precision[i - 1] + precision[i]) / 2.0
    return ap
