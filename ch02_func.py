from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSKeepFilter
from janome.tokenfilter import POSStopFilter

def get_words(s):
    a = Analyzer(token_filters=[POSStopFilter(['記号']),ExtractAttributeFilter('surface')])
    return list(a.analyze(s))

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