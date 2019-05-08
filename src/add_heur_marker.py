import codecs
import sys
import json
import nltk
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

def count_overlap(sent, sec_words):
    sent_words = set(sent.split())
    sent_words = sent_words - stopWords
    return len(list(sent_words & sec_words))

def word_set_from_sents(sent_list):
    res = set()
    for sent in sent_list:
        res.update(sent.split())
    return res

def add_parag_toks(abs_sents, art_text, tokenizer):
    art_words = [word_set_from_sents(sec) for sec in art_text]
    num_sec = len(art_words)

    new_sents = []
    # new_sents.append("<T> " + abs_sents[0])
    if num_sec == 1:
        # new_sents = new_sents + abs_sents[1:-1] + [abs_sents[-1]+" </T>"]
        new_sents = new_sents + abs_sents[1:]
        return new_sents

    cur_sec = 0
    for i, sent in enumerate(abs_sents):
        overlap_cur = count_overlap(sent, art_words[cur_sec])
        overlap_next = count_overlap(sent, art_words[cur_sec+1])
        if overlap_next >= overlap_cur:
            cur_sec += 1
            # new_sents[-1] = new_sents[-1] + " </T>"
            new_sents.append("<T> "+sent)
            if cur_sec == num_sec -1:
                new_sents = new_sents + abs_sents[i+1:] 
                break
        else:
            new_sents.append(sent)
    
    return new_sents


srcFile = sys.argv[1]
tgtFile = sys.argv[2]

print('Processing %s & %s ...' % (srcFile, tgtFile))
srcF = codecs.open(srcFile, "r", "utf-8")
tgtF = codecs.open(tgtFile, "w", "utf-8")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

while True:
    sline = srcF.readline()

    # normal end of file
    if sline == "":
        break
    
    sdict = json.loads(sline)
    abs_text = sdict['abstract_text']
    art_text = sdict['sections']
    new_abs_text = add_parag_toks(abs_text, art_text, tokenizer)
    sdict['abstract_text'] = new_abs_text
    json.dump(sdict, tgtF)
    tgtF.write("\n")
    # tgtF.write(json.dump(sdict)+"\n")

srcF.close()
tgtF.close()