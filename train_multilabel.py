from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import load_dataset, concatenate_datasets, DatasetDict
import sys
import re
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import collections
import random
import json

# logging
from transformers.utils.logging import WARNING as log_level
from transformers.utils.logging import set_verbosity as model_verbosity
from datasets.utils.logging import set_verbosity as data_verbosity
model_verbosity(log_level)
data_verbosity(log_level)

# remove warning from tokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# save on cache memory
from datasets import disable_caching
disable_caching()    #this stops cache for map()
CACHE = "/scratch/project_2002026/amanda/cache/"


# Hyperparameters
languages =  ["ar","bn","ca","en","es","eu","fr","hi","id","pt","sw","ur","vi","zh"]
LEARNING_RATE=7e-5
BATCH_SIZE=16
TRAIN_EPOCHS=8
MODEL_NAME = 'xlm-roberta-base'
PATIENCE = 3
threshold=0.5
# these are printed out by make_dataset.py
labels1 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP']#, 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels2 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP', 'LY', 'SP']

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', '--base_model', default=MODEL_NAME, metavar='HF-MODELNAME',
                    help='Pretrained model name')
    ap.add_argument('--data_name',type=str, metavar='HF-DATASETNAME', default="TurkuNLP/register_oscar",
                    help='Name of the dataset')
    ap.add_argument('--language',  type=json.loads,default=["en"], metavar='LIST-LIKE',
                    help='Language to be used from the dataset. Give as \'["en","zh"]\' ')
    ap.add_argument('--labels', type=json.loads, default=labels1,metavar='LIST-LIKE',
                    help='which labels to use, give as \'["IN","NA"]\' ')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=BATCH_SIZE, help='Batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=TRAIN_EPOCHS,
                    help='Number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=LEARNING_RATE, help='Learning rate')
    ap.add_argument('--patience', metavar='INT', type=int,
                    default=PATIENCE, help='Early stopping patience')
    ap.add_argument('--split', metavar='FLOAT', type=float, default=0.8,
                    help='Set train/val data split ratio, e.g. 0.8 to train on 80 percent')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=123, help='Random seed for splitting data')
    ap.add_argument('--downsample', metavar="INT", type=int,
                    default=None, help='downsample to 1/nth')
    ap.add_argument('--cache', default=CACHE, metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--checkpoints', default="checkpoints", metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='Path to save model to. {seed}_{languages}.pt added in the script')
    return ap



def get_label_counts(dataset, split):
    """ Calculates the frequencies of labels of a dataset. """
    label_counts = collections.Counter()
    for line in dataset[split]:
        for label in line['labels']:
            label_counts[label] += 1
    return label_counts


def resplit(dataset, options):
    temp = {}
    dataset_new = DatasetDict()
    keys = [i for i in dataset.keys()]
    if options.split>0:
        for k in keys:   # loop over languages => save the test split as separate key in result
            temp[k] = dataset[k].train_test_split(train_size=options.split, shuffle=True, seed=options.seed)
            dataset_new["validation_"+str(k)] = temp[k]["test"]
        dataset_new["train"] = concatenate_datasets([temp[k]["train"] for k in keys])
        dataset_new["validation"] = concatenate_datasets([temp[k]["test"] for k in keys])
        del temp
    else:
        dataset_new["train"] = concatenate_datasets([dataset[k] for k in keys])
    return dataset_new

def read_dataset(options):
    data_files = {str(i):str(i)+"/"+str(i)+"_00000.jsonl.gz" for i in options.language}
    print(f'reading {data_files} from {options.data_name}')
    try:
        dataset = load_dataset(options.data_name, data_files=data_files, cache_dir=options.cache)
    except FileNotFoundError:
        raise FileNotFoundError("Cannot read the dataset. Check that given language(s)="+str(options.language)+" exist(s) in the dataset.")
    if options.downsample is not None:
        suffix_dict = {1:"st", 2:"nd", 3:"rd"}
        n = options.downsample
        print(f'Filtering the data down to 1/{options.downsample}{suffix_dict.get(n%100 if (n%100)<20 else n%10, "th")}. Remove this step with --downsample=None.')
        dataset = dataset.filter(lambda example, idx: idx % options.downsample == 0, with_indices=True)
    return resplit(dataset, options)


#-------------------------------------preprocessing-----------------------------------------#

    
#def preprocess_text(d):
#    # Separate punctuations from words by whitespace
#    try:
#        d['text'] = re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", #d['text']))
#    except:
#        print("Warning: Unable to run regex on text of type", type(d['text']))
#    return d

def wrap_preprocess(options):
    def preprocess(d):
        d["labels"] = [i for i in d["labels"] if i in options.labels]
        d["text"] =re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", re.sub(r"([\.,:;\!\?\"\(\)])([\w\d])", r"\1 \2", d['text']))
        return d
    return preprocess


def binarize(dataset, options):
    """ Binarize the labels of the data. Fitting based on the whole data. """
    mlb = MultiLabelBinarizer()
    mlb.fit([options.labels])
    print("Binarizing the labels")
    dataset = dataset.map(lambda line: {'labels': mlb.transform([line['labels']])})
    return dataset, mlb.classes_


def wrap_tokenizer(tokenizer):
    def encode_dataset(d):
        try:
            output = tokenizer(d['text'], truncation= True, padding = True, max_length=512)
            return output
        except:     #for empty text
            output = tokenizer(" ", truncation= True, padding = True, max_length=512)
            return output

    return encode_dataset

def process_dataset(dataset, options):
    # filter empty and not-wanted labels out, add whitespace before punctuation
    print("Prepocessing text and labels")
    dataset = dataset.map(wrap_preprocess(options))
    dataset = dataset.filter(lambda example: example["labels"]!=[])
    # binarize
    dataset, mlb_classes = binarize(dataset, options)
    # update options to contain mappings for labels => this for some reason works for binarized labels
    options.label2id = dict(zip(mlb_classes, [i for i in range(0, len(mlb_classes))]))
    options.id2label = {v:k for k,v in options.label2id.items()}
    # tokenize
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(options.model_name, cache_dir=options.cache)
    dataset = dataset.map(wrap_tokenizer(tokenizer))
    return dataset, tokenizer

#--------------------------------------Training----------------------------------------#

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def multi_label_metrics(predictions, labels, threshold):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    # remove the unneccessary dimension 2 (shape is [num_examples, 1, num_labels])
    labels = np.reshape(labels, (-1,labels.shape[-1]))  
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_th05 = np.zeros(probs.shape)
    y_th05[np.where(probs >= 0.5)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    #roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average, # user-chosen or optimized threshold
               'f1_th05': f1_score(y_true=y_true, y_pred=y_th05, average='micro'), # report also f1-score with threshold 0.5
               #'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids,
        threshold=threshold)
    return result


def train(dataset, tokenizer, options):
    
    # Model downloading
    num_labels = len(options.labels)
    print("Downloading model", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(options.model_name, 
                                                               num_labels = num_labels,
                                                               label2id = options.label2id,
                                                               id2label = options.id2label,
                                                               cache_dir=options.cache)


    print("Initializing model", flush=True)
    train_args = TrainingArguments(
        output_dir=options.checkpoints,
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        learning_rate=options.lr,
        per_device_train_batch_size=options.batch_size,
        per_device_eval_batch_size=options.batch_size,
        num_train_epochs=options.epochs,
        gradient_accumulation_steps=4,
        save_total_limit=options.patience+1,
        disable_tqdm=False   ## True=disable progress bar in training
    )


    trainer = MultilabelTrainer(
        model,
        train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=options.patience)]
    )

    print("Training", flush=True)
    if options.epochs > 0:
        trainer.train()
    else:
        print("Epochs given <= 0, only evaluating")
    
    print("Evaluating", flush=True)

    reports=[]
    for key in options.language:
        # for classification report: get predictions
        val_predictions = trainer.predict(dataset["validation_"+str(key)])
        #predictions, true_labels, eval_results =trainer.predict(dataset["validation"])
        
        # apply sigmoid to predictions and reshape real labels
        p = 1.0/(1.0 + np.exp(- val_predictions.predictions))
        t = val_predictions.label_ids.reshape(p.shape)
        # apply treshold of 0.5
        pred_ones = [pl>0.5 for pl in p]
        true_ones = [tl==1 for tl in t]
    
        reports.append((key, classification_report(true_ones,pred_ones, target_names = options.label2id.keys())))
    return trainer.model, tokenizer, reports



if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    device = 'cuda' if cuda.is_available() else 'cpu'

    print("Reading data")
    dataset = read_dataset(options)
    print("Processing data")
    dataset, tokenizer = process_dataset(dataset, options)
    print("Ready to train:")
    model, _, reports = train(dataset, tokenizer, options)

    for k, r in reports:
        print(f'classification report for {k}:')
        print(r)
    # save the model
    if options.save_model is not None:
        lang = ''.join(options.language)
        options.save_model = options.save_model +"_"+ lang+ "_"+ str(options.seed)+".pt"
        torch.save(model, options.save_model)
