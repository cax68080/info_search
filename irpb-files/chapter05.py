from chapter01 import get_string_from_file
from chapter02 import get_words_from_file
from chapter03 import get_words, bows_to_cfs, load_aozora_corpus, get_bows,    add_to_corpus, get_weights, translate_bows, get_tfidfmodel_and_weights
from chapter04 import vsm_search


# Listing 5.3 #

def select_by_threshold(r, threshold=0.0):
    # rの長さ分の0を要素とするリストを作成
    answer = [0]*len(r)
    for i in r:
        # 類似度がthresholdより大きいときr[文書番号]を1にする
        if i[1] > threshold: answer[i[0]] = 1
    return answer


# Listing 5.5 #

from sklearn.metrics import precision_score, recall_score, f1_score

def print_scores(right_answer, my_answer):
    print('precision %.4f' % precision_score(right_answer, my_answer))
    print('recall %.4f' % recall_score(right_answer, my_answer))   
    print('f-measure %.4f' % f1_score(right_answer, my_answer))


# Listing 5.12 #

def top_n(r, n):
    answer = [0]*len(r)
    for i in range(n):
        answer[r[i]] = 1
    return answer


# Listing 5.14 #

def get_pr_curve(ranking, answer):
    # top_n(ranking, i)の適合率と再現率をそれぞれprecision[i]とrecall[i]へ
    # precision[0] = 1, recall[0] = 0とする
    precision = [1]
    recall = [0]
    for i in range(1, len(ranking) + 1):
        x = top_n(ranking, i)
        precision.append(precision_score(answer, x))
        recall.append(recall_score(answer, x))
    return precision, recall


# Listing 5.15 #
                                                             
import matplotlib.pyplot as plt

def draw_pr_curve(ranking, answer):
    precision, recall = get_pr_curve(ranking, answer)
    # グラフの描画範囲を設定
    plt.xlim(-0.05, 1.05)
    plt.ylim(0.0, 1.1)
    # 各軸のラベルを設定．x軸に再現率，Y軸に適合率
    plt.xlabel('recall')
    plt.ylabel('precision')
    # 曲線の下を塗りつぶす
    plt.fill_between(recall, precision, 0, facecolor='#FFFFCC')
    # 曲線を点と線で描く
    plt.plot(recall, precision, 'o-')


# Listing 5.18 #

def get_average_precision(ranking, answer):
    precision, recall = get_pr_curve(ranking, answer)
    ap = 0.0
    # (r[i-1], 0), (r[i-1], p[i-1]), (r[i], 0), (r[i], p[i]) で
    # 囲まれる面積をそれぞれap に加算
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i -1])*(precision[i -1] + precision[i])/2.0
    return ap

