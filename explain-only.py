import data_preparation
import explain_multilabel
from arguments import argparser   # argument parser in separate file
import sys
import numpy as np
import pandas as pd
import json
from torch import cuda
import torch
import spacy
import os
from transformers import AutoModelForSequenceClassification

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


def explain(options):

    print("Reading data",flush=True)
    dataset = data_preparation.read_dataset(options)
    print("Processing data",flush=True)
    dataset, tokenizer = data_preparation.process_dataset(dataset, options)
    print("Resulting dataset:")
    print(dataset)
    print(f"with label dictionary: {options.label2id}")
    print(f"Downloading --trained_model = {options.save_model}", flush=True)
    try:
        trained_model = AutoModelForSequenceClassification.from_pretrained(options.save_model)
    except:
        trained_model = torch.load(options.save_model)
    
    trained_model.to("cuda")
          

    # explain
    print("Explaining",flush=True)
    if options.parse_separately is not None and options.parser_model is not None:
        options.parser = spacy.load(options.parser_model)
        print(f'{options.parser_model} loaded from spacy.')
    explain_multilabel.explain_and_save_documents(dataset, trained_model, tokenizer, options)
        

if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    threshold=options.threshold
    os.makedirs(os.path.dirname(options.save_file), exist_ok=True)
    options.label2id = {k:v for k,v in zip(options.labels, range(0, len(options.labels)))}
    options.id2label = {v:k for k,v in options.label2id.items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    print(options)

    # begin
    with torch.device(device):
        explain(options)
