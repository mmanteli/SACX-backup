from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import data_preparation
import sys
import re
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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

# this is here because it needs to be set as global for trainer to see it
threshold=None



#---------------------------------Setting up Trainer------------------------------------#


class MultilabelTrainer(Trainer):
    """
    Overriding the Trainer's loss function to facilitate multilabel training.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()   # Binary Cross Entropy
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def multi_label_metrics(predictions, labels, threshold):
    """
    Calculate metrics that are applicable for multilabel classification.
    """
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
               'thr': threshold,
               'f1_th0.5': f1_score(y_true=y_true, y_pred=y_th05, average='micro'), # report also f1-score with threshold 0.5
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



#--------------------------------------Training----------------------------------------#


def train(dataset, tokenizer, options):
    """ 
    Function that downloads a base model, creates a tokenizer and a multilabel classifier based on it,
    and trains it. Returns the trained model, tokenizer and a classification report based on user defined threshold.
    """
    # force this
    global threshold
    threshold=options.threshold
    
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
        print("Epochs given <= 0, only evaluating", flush=True)
    
    print("Evaluating", flush=True)

    reports=[]
    for key in options.language:
        # for classification report: get predictions
        val_predictions = trainer.predict(dataset["validation_"+str(key)])
        #predictions, true_labels, eval_results =trainer.predict(dataset["validation"])
        
        # apply sigmoid to predictions and reshape real labels
        p = 1.0/(1.0 + np.exp(- val_predictions.predictions))
        t = val_predictions.label_ids.reshape(p.shape)
        # apply user threshold
        pred_ones = [pl>threshold for pl in p]
        true_ones = [tl==1 for tl in t]
    
        reports.append((key, classification_report(true_ones,pred_ones, target_names = options.label2id.keys())))
    return trainer.model, tokenizer, reports


# if you only want to run training, run python train_multilabel.py [--parameters]
if __name__=="__main__":

    # run this with 
    options = argparser().parse_args(sys.argv[1:])
    threshold=options.threshold
    device = 'cuda' if cuda.is_available() else 'cpu'

    print("Reading data", flush=True)
    dataset = read_dataset(options)
    print("Processing data",flush=True)
    dataset, tokenizer = process_dataset(dataset, options)
    print("Ready to train:",flush=True)
    model, _, reports = train(dataset, tokenizer, options)

    for k, r in reports:
        print(f'classification report for {k}:')
        print(r)
    # save the model
    if options.save_model is not None:
        lang = ''.join(options.language)
        options.save_model = options.save_model +"_"+ lang+ "_"+ str(options.seed)+".pt"
        torch.save(model, options.save_model)
