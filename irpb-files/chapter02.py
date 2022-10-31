from chapter01 import get_string_from_file


# Listing 2.6 #

from janome.tokenizer import Tokenizer

def get_m_snippet_from_file(filename, query, width=3):
    t = Tokenizer(wakati=True)
    qlist = list(t.tokenize(query))
    qlen = len(qlist)
    s = get_string_from_file(filename)
    slist = list(t.tokenize(s))
    for i in [k for k, v in enumerate(slist) if v == qlist[0]]:
        if qlist == slist[i:i + qlen]:
            return ''.join(slist[max(0,i - width):i + width + qlen])
    return None



# Listing 2.8 #

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordcloud(frequencies, font, width=600, height=400):
    wordcloud = WordCloud(background_color='white', font_path=font,
                          width=width, height=height)
    plt.figure(figsize=(width/50, height/50))
    plt.imshow(wordcloud.generate_from_frequencies(frequencies))
    plt.axis('off')
    plt.show()


# Listing 2.9 #

from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.tokenfilter import POSStopFilter

def get_words(s):
    a = Analyzer(token_filters=[POSStopFilter(['記号']), 
                                ExtractAttributeFilter('surface')])
    return list(a.analyze(s))

def get_words_from_file(f):
    return get_words(get_string_from_file(f))


# Listing 2.10 #

import matplotlib.font_manager as fm
from matplotlib import rcParams
    
japanese_font_candidates = ['Hiragino Maru Gothic Pro', 'Yu Gothic',
                            'Arial Unicode MS', 'Meirio', 'Takao',
                            'IPAexGothic', 'IPAPGothic', 'VL PGothic',
                            'Noto Sans CJK JP']

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
    fonts.append('font/ipaexg.ttf')
    return fonts

def configure_fonts_for_japanese(fonts=japanese_font_candidates):
    if hasattr(fm.fontManager, 'addfont'):
        fm.fontManager.addfont('font/ipaexg.ttf')
    else:
        ipa_font_files = fm.findSystemFonts(fontpaths='font')
        ipa_font_list = fm.createFontList(ipa_font_files)
        fm.fontManager.ttflist.extend(ipa_font_list)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = fonts


# Listing 2.12 #

import matplotlib.pyplot as plt

def plot_frequency(count, log_scale=False):
    y = list(sorted(count.values(), reverse=True))
    x = range(1, len(y) + 1)
    if log_scale:
        plt.loglog(x, y)
    else:
        plt.plot(x, y)

