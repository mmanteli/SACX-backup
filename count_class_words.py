from sklearn.feature_extraction.text import CountVectorizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import collections
import json
from train_multilabel import read_dataset
import sys
import re
from transformers import AutoTokenizer

""" Calculate term and document frequencies in corpus and save to file.

File 'class_df.json' is can be used in kws.py """


labels1 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP']
CACHE = "/scratch/project_2002026/amanda/cache/"

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_name',type=str, metavar='HF-DATASETNAME', default="TurkuNLP/register_oscar",
                    help='Name of the dataset')
    ap.add_argument('--language',  type=json.loads,default=["en"], metavar='LIST-LIKE',
                    help='Language to be used from the dataset. Give as \'["en","zh"]\' ')
    ap.add_argument('--labels', type=json.loads, default=labels1,metavar='LIST-LIKE',
                    help='which labels to use, give as \'["IN","NA"]\' ')
    ap.add_argument('--downsample', metavar="BOOL", type=bool,
                    default=True, help='downsample to 1/20th')
    ap.add_argument('--cache', default=CACHE, metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--seed', type=int, default=123)
    return ap


options = argparser().parse_args(sys.argv[1:])
options.split=0
dataset = read_dataset(options)
print(dataset)

class_DF = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
class_TF = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
vectorizer = CountVectorizer()
for field in dataset:
    for row in dataset[field]:
        labels, text = row["labels"],row["text"]
        labels = [l for l in labels if l in set(options.labels)]
        #text = re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", text))
        text = text.replace("\n"," ")
        try:
            m = vectorizer.fit([text])
            for word, cnt in zip(vectorizer.get_feature_names_out(), vectorizer.transform([text]).toarray()[0]):
                word = str(word)
                for label in labels:
                    class_TF[label][word] += int(cnt)
                    class_DF[label][word] += 1
        except ValueError:
            continue
        for label in labels:
            class_DF[label]['_N_DOCS'] += 1

json.dump(class_DF, open("class_df.json",'w'))
json.dump(class_TF, open("class_tf.json",'w'))
