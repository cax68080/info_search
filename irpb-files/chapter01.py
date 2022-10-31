# Listing 1.2 #

def check_query(filename, query):
    with open(filename, 'r', encoding='UTF-8') as f:
        s = f.read()
        return query in s

# Listing 1.9 #

import chardet

def get_string_from_file(filename):
    with open(filename, 'rb') as f:
        d = f.read()
        e = chardet.detect(d)['encoding']
        # 推定できなかったときはUTF-8で
        if e == None:
            e = 'UTF-8'
        return d.decode(e)


# Listing 1.10 #

def check_encoding_and_query(filename, query):
    s = get_string_from_file(filename)
    return query in s


# Listing 1.12 #

def get_ngram(string, N=1):
    return [string[i:i+N] for i in range(len(string) - N + 1)]


# Listing 1.13 #

from collections import Counter

def get_most_common_ngram(filename, N=1, k=1):
    s = get_string_from_file(filename)
    return Counter(get_ngram(s, N=N)).most_common(k)


# Listing 1.22 #

import re

def get_snippet_from_file(filename, query, width=2):
    s = get_string_from_file(filename)
    p = '.{0,%d}%s.{0,%d}' % (width, query, width)                              
    r = re.search(p, s)
    if r:
        return r.group(0)
    else:
        return None




