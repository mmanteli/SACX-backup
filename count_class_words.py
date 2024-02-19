from sklearn.feature_extraction.text import CountVectorizer
from arguments import argparser
import collections
import json
from data_preparation import read_dataset
from data_preparation import wrap_preprocess
import sys
import re
from transformers import AutoTokenizer

""" Calculate term and document frequencies in corpus and save to file.

File 'class_df.json' can be used in kws.py """



options = argparser().parse_args(sys.argv[1:])
options.split=0
print(options)
dataset = read_dataset(options)
print(dataset)
preprocess=wrap_preprocess(options)

if options.parse_separately is not None and options.parser_model is not None:
    import spacy
    parser = spacy.load(options.parser_model)

class_DF = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
class_TF = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
vectorizer = CountVectorizer(tokenizer=parser)
for field in dataset:
    for row in dataset[field]:
        #labels, text = row["labels"],row["text"]
        #labels = [l for l in labels if l in set(options.labels)]
        row = preprocess(row)
        labels, text = row["labels"],row["text"]
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

lang=''.join(options.language)
json.dump(class_DF, open(lang+"_class_df.json",'w'))
json.dump(class_TF, open(lang+"_class_tf.json",'w'))
