import train_multilabel
import explain_multilabel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
import pandas as pd
import json
from torch import cuda
import torch

# Hyperparameters
languages =  ["ar","bn","ca","en","es","eu","fr","hi","id","pt","sw","ur","vi","zh"]
LEARNING_RATE=7e-5
BATCH_SIZE=32
TRAIN_EPOCHS=5
MODEL_NAME = 'xlm-roberta-base'
PATIENCE = 3
threshold=0.5
int_bs=10
# these are printed out by make_dataset.py
labels1 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP']#, 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels2 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP', 'LY', 'SP']
CACHE = "/scratch/project_2002026/amanda/cache/"



def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--data_name', metavar='HF_DATASET', default="TurkuNLP/register_oscar",
                    help='Name of the dataset')
    ap.add_argument('--language', '--languages', type=json.loads, default=["en"], metavar='LIST-LIKE',
                    help='Language to be used from the dataset. Give as \'["en","zh"]\' ')
    ap.add_argument('--labels', type=json.loads, default=labels1, metavar='LIST-LIKE',
                    help='which labels to use, give as \'["IN","NA"]\' ')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=BATCH_SIZE, help='Batch size for training')
    ap.add_argument('--int_batch_size', metavar='INT', type=int, default=int_bs,
                    help='Batch size for integrated gradients')
    ap.add_argument('--epochs', metavar='INT', type=int, default=TRAIN_EPOCHS,
                    help='Number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=LEARNING_RATE, help='Learning rate')
    ap.add_argument('--patience', metavar='INT', type=int,
                    default=PATIENCE, help='Early stopping patience')
    ap.add_argument('--split', metavar='FLOAT', type=float, default=0.8,
                    help='Set train/val data split ratio, e.g. 0.8 to train on 80 percent')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=0, help='Random seed for splitting data')
    ap.add_argument('--downsample', metavar="INT", type=int,
                    default=None, help='downsample to 1/nth')
    ap.add_argument('--visualize', metavar="BOOL", type=bool,default=False,
                    help='If True, print HTML presentation. NOTE: this PRINTS, so redirect output using ">"!')
    ap.add_argument('--checkpoints', default='checkpoints', metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--cache', default=CACHE, metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='Path to save the model to. "{seed}_{lang}.pt" added in the script.')
    ap.add_argument('--save_file', default="./explanations/exp", metavar='FILE',
                    help='Path to save file. "{seed}_{lang}.tsv" added in the script.')
    ap.add_argument('--save_reports', default=None, metavar='FILE',
                    help='path where to save the results to. "{seed}_{lang}_report.txt" added in the script')
    return ap



def train_and_explain(options):

    # training
    print("Reading data")
    dataset = train_multilabel.read_dataset(options)
    print("Processing data")
    dataset, tokenizer = train_multilabel.process_dataset(dataset, options)
    print("Ready to train:")
    trained_model, tokenizer, reports = train_multilabel.train(dataset, tokenizer, options)
        
    if options.save_reports is not None:
        for i,k in enumerate(options.language):
            full_path = options.save_reports+"_"+str(options.seed)+ "_"+''.join(k)+"_report.txt"
            with open(full_path, 'w') as f:
                f.write(reports[i][1])
    if options.save_model is not None:
        lang = ''.join(options.language)
        options.save_model = options.save_model+"_"+str(options.seed)+"_"+lang+".pt"
        torch.save(trained_model, options.save_model)

    # explain
    print("Explaining")
    explain_multilabel.explain_and_save_documents(dataset, trained_model, tokenizer, options)
    

if __name__=="__main__":
        
    options = argparser().parse_args(sys.argv[1:])
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(options)
    train_and_explain(options)
