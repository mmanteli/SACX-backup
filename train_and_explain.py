import data_preparation
import train_multilabel
import explain_multilabel
from arguments import argparser   # argument parser in separate file
import sys
import numpy as np
import pandas as pd
import json
from torch import cuda
import torch
import spacy

# logging
from transformers.utils.logging import WARNING as log_level
from transformers.utils.logging import set_verbosity as model_verbosity
from datasets.utils.logging import set_verbosity as data_verbosity
model_verbosity(log_level)
data_verbosity(log_level)

# remove warning from tokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# this needs to be a global variable for it to work, it will be set to options.threshold
threshold=None


def train_and_explain(options):

    # training
    print("Reading data",flush=True)
    dataset = data_preparation.read_dataset(options)
    print("Processing data",flush=True)
    dataset, tokenizer = data_preparation.process_dataset(dataset, options)
    print("Ready to train:",flush=True)
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
    print("Explaining",flush=True)
    if options.parse_separately is not None and options.parser_model is not None:
        options.parser = spacy.load(options.parser_model)
        print(f'{options.parser_model} loaded from spacy.')
    explain_multilabel.explain_and_save_documents(dataset, trained_model, tokenizer, options)
    

if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    threshold=options.threshold
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(options)

    # begin
    train_and_explain(options)
