import sys
import os
import numpy as np
import codecs
import logging
from SARI import SARIsent
from nltk.translate.bleu_score import *
smooth = SmoothingFunction()
from nltk import word_tokenize
from textstat.textstat import textstat
import Levenshtein
import nltk
from nltk.tokenize import RegexpTokenizer
import syllables_en

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
SPECIAL_CHARS = ['.', ',', '!', '?']
logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)


def get_words(text=''):
    words = TOKENIZER.tokenize(text)
    filtered_words = []
    for word in words:
        if word in SPECIAL_CHARS or word == " ":
            pass
        else:
            new_word = word.replace(",","").replace(".","")
            new_word = new_word.replace("!","").replace("?","")
            filtered_words.append(new_word)
    return filtered_words

def get_sentences(text=''):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences

def count_syllables(words):
    syllableCount = 0
    for word in words:
        syllableCount += syllables_en.count(word)
    return syllableCount

def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def folders_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,f)) ]

def files_in_folder_only(mypath):
    return [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def remove_features(sent):
    tokens = sent.split(" ")
    return " ".join([token.split("|")[0] for token in tokens])

def remove_underscores(sent):
    return sent.replace("_", " ")

def replace_parant(sent):
    sent = sent.replace("-lrb-", "(").replace("-rrb-", ")")
    return sent.replace("(", "-lrb-").replace(")", "-rrb-")

def lowstrip(sent):
    return sent.lower().strip()

def normalize(sent):
    return replace_parant(lowstrip(sent))

def as_is(sent):
    return sent

def get_hypothesis(filename):
    hypothesis = '-'
    if "_h1" in filename:
        hypothesis = '1'
    elif "_h2" in filename:
        hypothesis = '2'
    elif "_h3" in filename:
        hypothesis = '3'
    elif "_h4" in filename:
        hypothesis = '4'
    return hypothesis

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def print_scores(pairs, whichone = ''):
    # replace filenames by hypothesis name for csv pretty print
    for k,v in pairs:
        hypothesis = get_hypothesis(k)
        print("\t".join( [whichone, "{:10.2f}".format(v), k, hypothesis] ))

def SARI_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    for src, pred, ref in zip(*files):
        references = [preprocess(r) for r in ref.split('\t')]
        scores.append(SARIsent(preprocess(src), preprocess(pred), references))
    for fis in files:
        fis.close()
    return mean(scores)


# BLEU doesn't need the source
def BLEU_file(source, preds, refs, preprocess=as_is):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [preds, refs]]
    scores = []
    references = []
    hypothese = []
    for pred, ref in zip(*files):
        references.append([word_tokenize(preprocess(r)) for r in ref.split('\t')])
        hypothese.append(word_tokenize(preprocess(pred)))
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return corpus_bleu(references, hypothese, smoothing_function=smooth.method3)


def worddiff_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    worddiff = 0
    n = 0
    for src, pred in zip(*files):
        source = word_tokenize(preprocess(src))
        hypothese = word_tokenize(preprocess(pred))
        n += 1
        worddiff += len(source) - len(hypothese)

    worddiff /= float(n)
    for fis in files:
        fis.close()

    return worddiff / 100.0


def IsSame_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    issame = 0
    n = 0.
    for src, pred in zip(*files):
        source = preprocess(src)
        hypothese = preprocess(pred)
        n += 1
        issame += source == hypothese

    issame /= n
    for fis in files:
        fis.close()

    return issame / 100.0


def FKGL_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    score = 0
    n = 0.
    for src, pred in zip(*files):
        hypothese = preprocess(pred)
        words = get_words(hypothese)
        word_count = float(len(words))
        sentence_count = float(len(get_sentences(hypothese)))
        syllable_count = float(count_syllables(words))
        score += 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        n += 1

    score /= n
    for fis in files:
        fis.close()

    return round(score, 2) / 100


def FKdiff_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    fkdiff = 0
    n = 0.
    for src, pred in zip(*files):
        # hypothese = preprocess(pred)
        # source = preprocess(src)
        hypothese = (pred)
        source = (src)
        # print(source)
        # print(hypothese)

        fkdiff += (textstat.flesch_reading_ease(hypothese) - textstat.flesch_reading_ease(source))
        n += 1
        # fkdiff= 1/(1+np.exp(-fkdiff))

    fkdiff /= n
    for fis in files:
        fis.close()

    return fkdiff / 100.0


def LD_file(source, preds, refs, preprocess):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    LD = 0
    n = 0.
    for src, pred in zip(*files):
        hypothese = preprocess(pred)
        source = preprocess(src)
        LD += Levenshtein.distance(hypothese, source)
        n += 1

    LD /= n
    for fis in files:
        fis.close()

    return LD / 100.0


def score(source, refs, fold, METRIC_file, preprocess=as_is):
    # new_files = files_in_folder(fold)
    data = []
    for fis in fold:
        # ignore log files
        if ".log" in os.path.basename(fis):
            continue
        logging.info("Processing "+os.path.basename(fis))
        val = 100*METRIC_file(source, fis, refs, preprocess)
        logging.info("Done "+str(val))
        data.append((os.path.basename(fis), val))
    data.sort(key=lambda tup: tup[1])
    data.reverse()
    return data, None


def map_to_array(score_dict):
    def get_beam_order_from_filename(filename):
        filename = filename.split('_')
        beam = int(filename[2][1:])
        hyp_order = int(filename[3][1])
        return beam, hyp_order, filename[1]

    score_arr_dict = {}
    for filename, val in score_dict:
        try:
            beam, hyp_order, subset = get_beam_order_from_filename(filename)
        except:
            beam, hyp_order, subset = 5, 1, 'test'
        if subset in score_arr_dict:
            score_arr_dict[subset][beam-5, hyp_order-1] = round(val, 2)
        else:
            score_arr_dict[subset] = np.zeros((8, 5))
            score_arr_dict[subset][beam - 5, hyp_order - 1] = round(val, 2)
    return score_arr_dict


if __name__ == '__main__':
    try:
        source = sys.argv[1]
        logging.info("Source: " + source)
        refs = sys.argv[2]
        logging.info("References in tsv format: " + refs)
        pred_path = sys.argv[3]
        logging.info("Path of predictions: " + pred_path)
    except:
        logging.error("Input parameters must be: " + sys.argv[0]
            + "    SOURCE_FILE    REFS_TSV (paste -d \"\t\" * > reference.tsv)    DIRECTORY_OF_PREDICTIONS")
        sys.exit(1)

    '''
        SARI can become very unstable to small changes in the data.
        The newsela turk references have all the parantheses replaced
        with -lrb- and -rrb-. Our output, however, contains the actual
        parantheses '(', ')', thus we prefer to apply a preprocessing
        step to normalize the text.
    '''
    preds = open(pred_path, 'r').readlines()
    fold = []
    for idx in range(4):
        preds_tmp = preds[idx::4]
        filename_tmp = pred_path+'_h{}'.format(idx+1)
        fold.append(filename_tmp)
        open(filename_tmp, 'w').write(''.join(preds_tmp))

    sari_test, sari_arr = score(source, refs, fold, SARI_file, normalize)
    bleu_test, bleu_arr = score(source, refs, fold, BLEU_file, lowstrip)
    worddiff_test, worddiff_arr = score(source, refs, fold, worddiff_file, lowstrip)
    FKdiff_test, FKdiff_arr = score(source, refs, fold, FKdiff_file, lowstrip)
    IsSame_test, IsSame_arr = score(source, refs, fold, IsSame_file, lowstrip)
    LD_test, LD_arr = score(source, refs, fold, LD_file, lowstrip)
    FKGL_test, FKGL_arr = score(source, refs, fold, FKGL_file, lowstrip)

    # whichone = os.path.basename(os.path.abspath(os.path.join(fold, '..'))) + \
    #                 '\t' + \
    #                 os.path.basename(refs).replace('.ref', '').replace("test_0_", "")
    # print_scores(sari_test, "SARI\t" + whichone)
    # print_scores(bleu_test, "BLEU\t" + whichone)

    # print('\nSARI:')
    # for key, val in sari_arr.items():
    #     print(key, val)
    # print('\nBLEU:')
    # for key, val in bleu_arr.items():
    #     print(key, val)
    # print('\nWORD DIFF:')
    # for key, val in worddiff_arr.items():
    #     print(key, val)
    # print('\nFK DIFF:')
    # for key, val in FKdiff_arr.items():
    #     print(key, val)
    # print('\nLD:')
    # for key, val in LD_arr.items():
    #     print(key, val)
    # print('\nIsSame Percent:')
    # for key, val in IsSame_arr.items():
    #     print(key, val)
    # print('\nFKGL:')
    # for key, val in FKGL_arr.items():
    #     print(key, val)