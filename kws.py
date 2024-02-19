import numpy as np
import csv
import glob
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from numpy.core.defchararray import title
from numpy.core.numeric import NaN
import pandas as pd
import warnings
import json
import matplotlib.pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")

csv.field_size_limit(sys.maxsize)


# HYPERPARAMETRES
WORDS_PER_DOC = 100
NUMBER_OF_KEYWORDS = 40
SELECTION_FREQ = 0.6
STD_THRESHOLD = 0.2
MIN_WORD_FREQ = 3
PREDICTION_THRESHOLD = 0.5
SAVE_N = 100
QUANTILE = 0.25
SAVE_FILE = "keywords/"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data', metavar='FILE', required=True,
                    help='Path to results of explanation. Give "globbed" e.g. explanations/*.tsv.')
    ap.add_argument('--words_per_doc', metavar='INT', type=int, default=WORDS_PER_DOC,
                    help='Number of best words chosen per each doc-label pair. Optimize for your needs.')
    ap.add_argument('--filter', metavar='FILE', default = 'selectf', choices=["selectf", "std"],
                    help='Method for filtering: std or selection frequency.')
    ap.add_argument('--selection_freq', metavar='FLOAT', type=float, default=SELECTION_FREQ, 
                    help='Threshold for filtering: % of how many rounds the word must be present in order to be selected. Optimize')
    ap.add_argument('--std_threshold', metavar='FLOAT', type=float,
                    default=STD_THRESHOLD, help='Threshold for filtering: how much can the score a word gets fluctuate inside one label.')
    ap.add_argument('--min_word_freq', metavar='INT', type=int, default=MIN_WORD_FREQ,
                    help='Threshold for dropping words that are too rare in the corpus. For this, provide corpus information')
    ap.add_argument('--corpus_information', metavar='FILE', type=str, default=None,
                    help='json file containing frequencies of words in corpus for each label.')
    ap.add_argument('--save_n', metavar='INT', type=int, default=SAVE_N,
                    help='How many extracted words per class are saved.')  #"This needs to be really high, explanation in comments" where though
    ap.add_argument('--save_file', default=SAVE_FILE, metavar='FILE',
                    help='path and beginning of filename where the results are saved. "{label}_stable.csv" added in the script.')
    ap.add_argument('--style', metavar='STR', type=str, default="all", choices=["exact","contains", "all", "false_classf"],
                    help='exact: correct label is the predicted label (usefull in multiclass clsf) \n \
                    contains: correct label contains predicted label \n \
                    all: get all keywords wrt prediction \n \
                    false_classf: keywords extracted from falsely classified documents, i.e, correct label does not contain predicted label.')
    ap.add_argument('--prediction_th', type=float, default=PREDICTION_THRESHOLD,
                    help='How confident the prediction should be, 0...1.')
    return ap

def preprocess(data, options):
    """
    Remove errors in data.
    """
    data['token'] = data['token'].str.lower()
    data['token'] = data['token'].str.replace('[^\w\s]','')
    data.replace("", NaN, inplace=True)   # for empty words, removed right next below
    data.dropna(axis=0, how='any', inplace=True)
    data = data[data.pred != "None"]
    # remove documents with uncertain predictions TODO: This will work once explain.py saves in the right format
    #data["certainty"] = data['probs'].apply(lambda x: max(eval(x)))
    #data = data[data.certainty >= options.prediction_th]
    return data[['id', 'label', 'pred', 'token', 'score']]

def read_data_csv(data_name, options, delim="\t"):
    """ read the data from a csv and remove null values (no predictions)"""
    data = pd.read_csv(data_name, delimiter = delim)
    data = preprocess(data, options)
    return data

def read_data_other_format(data_name,options, delimiter="\t"):
    full = []
    for line in open(data_name, "r", encoding="utf-8"):
        new = []
        line=line.strip().split(delimiter)
        assert len(line)==6, "Given data has wrong number of fields per line, give ['id','label','pred','token','score','probs']."
        new.append(**line)
        full.append(new)
    data=pd.DataFrame(full,  columns = ['id', 'label', 'pred', 'token', 'score', 'probs'])
    data=preprocess(data, options)
    return data
        
def choose_n_best(data, n):
    """ choose n best scoring words per document """
    df_new = data.sort_values('score', ascending=False).groupby(['id', 'pred']).head(n)
    df_new.sort_index(inplace=True)
    return df_new

def class_frequencies(data):
    """
    Calculate mean, std of scores and number of sources, and add them to the dataframe
    """
    l = data.groupby(['token','pred'])['label'].unique()
    a = data.groupby(['token','pred'])['score'].unique()
    b = data.groupby(['token','pred'])['score'].mean()
    b2 = data.groupby(['token','pred'])['score'].std(ddof=0)
    c = data.groupby(['token','pred'])['source'].unique()

    return pd.concat([l.to_frame(), a.to_frame(), b.to_frame().add_suffix("_mean"), b2.to_frame().add_suffix("_std"), c.to_frame()],axis=1)


def flatten(t):
    """
    Flattens a nested list to a normal list
    [[2,3], 4] = [2,3,4]
    """
    return [item for sublist in t for item in sublist]



def read_files(options):
    df_list = []
    num_files = 0

    # read all the files. options.data should be in format /in/this/dir/*{something}.tsv or similar
    for filename in glob.glob(options.data):
        #try:
            if '.tsv' in filename:
                df = read_data_csv(filename, options, delim="\t")
                print(f'file {filename} read succesfully.', flush=True)
            elif '.csv' in filename:
                df = read_data_csv(filename, options, delim=",")
                print(f'file {filename} read succesfully.', flush=True)
            else:
                df = read_data_other_format(filename, options)
                print(f'file {filename} read succesfully.', flush=True)
            num_files +=1
        #except:
            #print(f'Error reading {filename}', flush=True)
            #continue
        
            # for each document-label pair, choose best scoring tokens
            df = choose_n_best(df, options.words_per_doc)
            # add a source tag for each;
            df['source'] = "".join(filename.split("/")[-1])
            # collect it
            df_list.append(df)

    # concatenate all for further analysis
    # concatenation faster if done all at once; that's why we're creating a list.
    assert num_files > 0, "No files could be read succesfully"
    df_full = pd.concat(df_list)
    del df_list
    return df_full, num_files

def filter_with_corpus_insights(data, options):
    """
    Take out everything that has a low document frequency wrt current label (key, number).
    This also removes all tokenization errors, since their document frequency
    should always be 0.
    Save to a file.
    """
    with open(options.corpus_information) as f: 
        frequencies = json.load(f)

    data["corpus_freq"] = data.apply(lambda x: frequencies[eval(x.pred)[0]][x.token] if x.token in frequencies[eval(x.pred)[0]] else 0, axis=1)

    data = data[data.corpus_freq > options.min_word_freq]

    return data

def extract_keywords(options):
    
    # read all files with id, token, label, pred, score, logits
    df_full, num_files = read_files(options)
    df_full["pred"].apply(lambda x: list(map(str, eval(x))))
    df_full["label"].apply(lambda x: list(map(str, eval(x))))
    labels = df_full.pred.unique()
    print("Found labels are: ", labels)
    # Filter out tokens without any letters => remove years, punctuation, quotation marks... TODO: maybe before choose n??
    df_full = df_full[df_full['token'].apply(lambda x: any([y.isalpha() for y in x]))] 

    # limit the analysis for correct classifications or false classifications (default: no limitation):
    if options.style=="exact":
        df_full = df_full[df_full.apply(lambda x: x.pred==x.label, axis=1)]
    elif options.style=="contains":
        df_full = df_full[df_full.apply(lambda x: x.pred[1:-2] in x.label, axis=1)]
    elif options.style=="false_classf":
        df_full = df_full[df_full.apply(lambda x: x.pred[1:-2] not in x.label, axis=1)]
    # Next, get statistic of scores for each word-prediction pair
    # for example:
    # token       label   score        source
    # mouse       [5]     [0.5]        [file2]
    # mouse       [5]     [0.4]        [file4]
    # mouse       [3]     [0.2]        [file2]
    # turns into
    # token       label   score            score_mean   score_std     source
    # mouse       [5]     [0.5, 0.4]       [0.45]       [0.05]        [file2, file4]
    # mouse       [3]     [0.2]            [0.2]        [0.00]        [file2]

    # do what is described above
    freq_array = class_frequencies(df_full)
    # map to contain the number of times selected
    freq_array["source_number"] = freq_array['source'].apply(lambda x: len(x))

    print("Filtering", flush=True)
    if options.filter == 'std':
        # we filter out everything where the score_std is higher than the threshold
        df_save = freq_array[freq_array.score_std < options.std_threshold]
        df_unstable = freq_array[freq_array.score_std >= options.std_threshold]
    elif options.filter == 'selectf':
        # here we look at the selection frequency e.g. how many separate sources the word is in
        # and drop if it is in less than the specified threshold (fraction)
        df_save = freq_array[freq_array.source_number >= options.selection_freq*num_files]
        df_unstable = freq_array[freq_array.source_number < options.selection_freq*num_files]

    # sort the values by label and mean score, and take a certain amount of best results per label
    print("Sorting", flush=True)
    df_save = df_save.sort_values(['pred','score_mean'], ascending=[True, False]).groupby(['pred'],as_index=False).head(options.save_n)
    df_unstable = df_unstable.sort_values(['pred','score_mean'], ascending=[True, False]).groupby(['pred'], as_index=False).head(options.save_n)


    # drop info that is no longer needed
    df_save.drop(['score','source'], axis=1, inplace=True)
    df_unstable.drop(['score','source'], axis=1, inplace=True)

    # FLATTEN as pandas group_by makes querying impossible
    df_save= df_save.reset_index()
    df_unstable= df_unstable.reset_index()

    print("Saving unstable keywords...", flush=True)
    # save unstable words
    df_unstable.to_csv(options.save_file+"unstable.csv")

    if options.corpus_information!=None:
        df_save = filter_with_corpus_insights(df_save, options)
    
    print("Saving stable keywords...", flush=True)
    #df_save.to_csv("stable.csv")
    for i in labels:
        df_save[df_save.pred==i].to_csv(options.save_file+str(eval(i)[0])+"_stable.csv")
        
    print("Everything done", flush=True)


if __name__=="__main__":
    #print("kws.py",flush = True)
    options = argparser().parse_args(sys.argv[1:])
    print(options, "\n",flush = True)
    extract_keywords(options)
