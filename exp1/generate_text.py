import sys
from itertools import chain
import sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import codecs

import MeCab
import pycrfsuite
from pprint import pprint
import csv
import mojimoji
import re

def is_hiragana(ch):
    return 0x3040 <= ord(ch) <= 0x309F

def is_katakana(ch):
    return 0x30A0 <= ord(ch) <= 0x30FF

def get_character_type(ch):
    if ch.isspace():
        return 'ZSPACE'
    elif ch.isdigit():
        return 'ZDIGIT'
    elif ch.islower():
        return 'ZLLET'
    elif ch.isupper():
        return 'ZULET'
    elif is_hiragana(ch):
        return 'HIRAG'
    elif is_katakana(ch):
        return 'KATAK'
    else:
        return 'OTHER'

def get_character_types(string):
    character_types = map(get_character_type, string)
    character_types_str = '-'.join(sorted(set(character_types)))

    return character_types_str

def extract_pos_with_subtype(morph):
    idx = morph.index('*')

    return '-'.join(morph[1:idx])

def word2features(sent, i):
    word = sent[i][0]
    chtype = get_character_types(sent[i][0])
    postag = extract_pos_with_subtype(sent[i])
    features = [
        'bias',
        'word=' + word,
        'type=' + chtype,
        'postag=' + postag,
    ]
    if i >= 2:
        word2 = sent[i-2][0]
        chtype2 = get_character_types(sent[i-2][0])
        postag2 = extract_pos_with_subtype(sent[i-2])
        iobtag2 = sent[i-2][-1]
        features.extend([
            '-2:word=' + word2,
            '-2:type=' + chtype2,
            '-2:postag=' + postag2,
            '-2:iobtag=' + iobtag2,
        ])
    else:
        features.append('BOS')

    if i >= 1:
        word1 = sent[i-1][0]
        chtype1 = get_character_types(sent[i-1][0])
        postag1 = extract_pos_with_subtype(sent[i-1])
        iobtag1 = sent[i-1][-1]
        features.extend([
            '-1:word=' + word1,
            '-1:type=' + chtype1,
            '-1:postag=' + postag1,
            '-1:iobtag=' + iobtag1,
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        chtype1 = get_character_types(sent[i+1][0])
        postag1 = extract_pos_with_subtype(sent[i+1])
        features.extend([
            '+1:word=' + word1,
            '+1:type=' + chtype1,
            '+1:postag=' + postag1,
        ])
    else:
        features.append('EOS')

    if i < len(sent)-2:
        word2 = sent[i+2][0]
        chtype2 = get_character_types(sent[i+2][0])
        postag2 = extract_pos_with_subtype(sent[i+2])
        features.extend([
            '+2:word=' + word2,
            '+2:type=' + chtype2,
            '+2:postag=' + postag2,
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [morph[-1] for morph in sent]


def sent2tokens(sent):
    return [morph[0] for morph in sent]



m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
f = open('line.csv', 'r')
out = open('tweet_tagged.txt', 'w')
dataReader = csv.reader(f)
for row in dataReader:
    text = row[0]
    text = mojimoji.zen_to_han(text, kana=False)
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    text = re.sub(r'[-~〰−―]', '〜', text)
    text = re.sub(r'[「【『［〈《]', '[', text)
    text = re.sub(r'[」】』］〉》]', ']', text)
    text = re.sub(r'[（]', '(', text)
    text = re.sub(r'[）]', ')', text)
    text = re.sub(r'[／]', '/', text)
    parsed = m.parse(text)

    mecab_parsed = []
    for line in parsed.split('\n'):
        line_array = line.split('\t')
        if len(line_array) < 2:
            break

        mecab_item = []
        mecab_item.append(line_array[0])
        mecab_item.extend(line_array[1].split(','))
        mecab_parsed.append(mecab_item)

    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')
    tagged = tagger.tag(sent2features(mecab_parsed))

    for i in range(len(mecab_parsed)):
        mecab_parsed[i].append(tagged[i])

    for mecab_item in mecab_parsed:
        out.write('\t'.join(mecab_item) + '\n')

    out.write('\n')

out.close()
