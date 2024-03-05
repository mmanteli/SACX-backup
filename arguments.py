# MASTER ARGUMENTS
# you can change them here or on command line
# Example usage for bash:
# ---------------------------------------------------- #
"""
#!/bin/bash

# Loop over 10 random seed values, you can choose also a higher number
for ((i=1; i<=10; i++)); do
    # Generate a random seed value
    seed=$((RANDOM + 1))

    # Run the program with the random seed
    python3 train_and_explain --seed $seed  # add other parameters here if needed
done
"""
# ---------------------------------------------------- #
# with slurm it is recommended that you sbatch the jobs separately.
# depending on the dataset size the runtime is 20min to 4 hours



from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import sys


MODEL_NAME = 'xlm-roberta-base'
DATA_NAME = "TurkuNLP/register_oscar" # or path; if path, list-like or separate by comma
DATA_TYPE = "oscar"
DELIMITER = '\t'
LANGUAGE = ["en"]  # Can have multiple, if DATA_NAME is a path, give equal amount of languages as paths
labels1 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP']#, 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'mt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels2 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP', 'LY', 'SP']
threshold=0.5   # prediction threshold
VISUALIZE = False    # if True, prints, so redirect output to a .html file
PARSE_SEPARATELY = None  # similar to language
PARSER_MODEL = None  # similar to data_name, as a list or separate by comma
LEARNING_RATE=7e-5
BATCH_SIZE=16
TRAIN_EPOCHS=8
SPLIT = 0.8
PATIENCE = 3
INT_BS=8    # not implemented really
DOWNSAMPLE = None
CACHE = "cache"
CHECKPOINTS = "checkpoints"
SAVE_TRAINED_MODEL = None
SAVE_REPORTS = None
SAVE_EXPLANATIONS = "./explanations/exp"
# SEED <= This is what you should change on command line to create different models, e.g. run python train_and_explain.py --seed=0, then --seed=1, etc.


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', '--base_model', default=MODEL_NAME, metavar='HF-MODELNAME',
                    help='Pretrained model name')
    ap.add_argument('--data_name','--data_path',type=str, metavar='HF-DATASETNAME or PATH', default=DATA_NAME,
                    help='Name of the dataset or path to a dataset.')
    ap.add_argument('--data_type', metavar='data_type', default=DATA_TYPE,
                    help='Type of the dataset; local "csv" or "huggingface" dataset.')
    ap.add_argument('--delimiter', metavar="CHARACTER", default=DELIMITER,
                    help="which delimiter in data, if applicable.")
    ap.add_argument('--language',  type=json.loads, metavar='LIST-LIKE', default=LANGUAGE,
                    help='Language to be used from the dataset, if applicable. Give as \'["en","zh"]\'. ')
    ap.add_argument('--labels', type=json.loads, metavar='LIST-LIKE', default=labels1,
                    help='which labels to use, give as \'["IN","NA"]\'. Others discarded. ')
    ap.add_argument('--threshold', type=float, metavar='FLOAT', default=threshold,
                    help='Prediction threshold for the classifier. NOT TESTED.')
    ap.add_argument('--visualize', metavar="BOOL", type=bool, default=VISUALIZE,
                    help='If True, print a HTML presentation. NOTE: this PRINTS, so redirect output using ">" to a file ending in .html !')
    ap.add_argument('--parse_separately', metavar='STRING/LANGUAGE', default=PARSE_SEPARATELY,
                    help='Language where separate parser is used (e.g. Chinese or Korean).')
    ap.add_argument('--parser_model', metavar='SPACY MODEL', default=PARSER_MODEL,
                   help='Model to do the separate parsing.')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=BATCH_SIZE, help='Batch size for training.')
    ap.add_argument('--int_batch_size', metavar='INT', type=int, default=INT_BS,
                    help='Batch size for integrated gradients (explanation). DOES NOT WORK!')
    ap.add_argument('--epochs', metavar='INT', type=int, default=TRAIN_EPOCHS,
                    help='Number of training epochs.')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=LEARNING_RATE, help='Learning rate.')
    ap.add_argument('--patience', metavar='INT', type=int,
                    default=PATIENCE, help='Early stopping patience.')
    ap.add_argument('--split', metavar='FLOAT', type=float, default=SPLIT,
                    help='Set train/val data split ratio, e.g. 0.8 to train on 80 percent.')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=123, help='Random seed for splitting data.')
    ap.add_argument('--downsample', metavar="INT", type=int,
                    default=DOWNSAMPLE, help='downsample data to 1/nth.')
    ap.add_argument('--cache', default=CACHE, metavar='FILE',
                    help='Save model checkpoints to directory.')
    ap.add_argument('--checkpoints', default=CHECKPOINTS, metavar='FILE',
                    help='Save model checkpoints to directory.')
    ap.add_argument('--save_model', '--trained_model', default=SAVE_TRAINED_MODEL, metavar='FILE',
                    help='Path to which the trained model is saved and from which it is loaded from. {seed}_{languages}.pt added in the script.')
    ap.add_argument('--save_reports', default=SAVE_REPORTS, metavar='FILE',
                    help='path where to save the results to. "{seed}_{lang}_report.txt" added in the script')
    ap.add_argument('--save_file','--explanations', default=SAVE_EXPLANATIONS, metavar='FILE',
                    help='Path to save file. {seed}_{language}.tsv added in the script.')
    return ap


# testing... no need to run this for other purposes!
if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    print(options)
    


