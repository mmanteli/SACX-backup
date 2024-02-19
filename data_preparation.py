# this contains the different ways to read a dataset in addition to preprocessing.
from arguments import argparser
import re
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import disable_caching
disable_caching()    #this stops cache for map()
# logging
from transformers.utils.logging import WARNING as log_level
from transformers.utils.logging import set_verbosity as model_verbosity
from datasets.utils.logging import set_verbosity as data_verbosity
model_verbosity(log_level)
data_verbosity(log_level)

#----------------------------------------reading--------------------------------------------#


def get_label_counts(dataset, split):
    """ Calculates the frequencies of labels of a dataset. """
    label_counts = collections.Counter()
    for line in dataset[split]:
        for label in line['labels']:
            label_counts[label] += 1
    return label_counts


def resplit(dataset, options):
    """ creates mixed and separate language splits for training. See read_csv() docstring. """
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

def read_csv(options):
    """
    Looks at options-attributes 'language' and 'data_path/data_name'. Constructs a dataset based on these.
    Use options.delimiter to change between csv, tsv, etc.
    E.g. language = ["jp","zh"], data_name = path_to_jp,path_to_zh
    =>
    dataset(
        train: {combined train of jp+zh}
        validation: {combined validation of jp+zh}
        validation_jp : {validation of jp}
        validation_zh : {validation of zh}
    )
    """
    data_files={}
    path = options.data_name.replace(" ", "").split(",")
    assert len(options.language)==len(path), f'Given languages and number of data paths do not align. Languages={options.language}, paths={path}.'
    for l,d in zip(options.language,path):
        data_files[l] = d
    print(data_files)
    dataset = load_dataset(
        'csv',
        data_files=data_files,
        delimiter=options.delimiter,
        column_names=['labels', 'text'],
        cache_dir = options.cache
        )
    return dataset

def read_huggingface(options):
    print(f'reading {options.data_name}')
    try:
        dataset = load_dataset(options.data_name, cache_dir=options.cache)
    except FileNotFoundError:
        raise FileNotFoundError("""Cannot read the dataset. Check that given language(s)="+str(options.language)+" exist(s) in the dataset,
        or specify --data_type='csv' if you're trying to read a local file.""")
    return dataset

def read_oscar(options):
    data_files = {str(i):str(i)+"/"+str(i)+"_00000.jsonl.gz" for i in options.language}
    print(f'reading {data_files} from {options.data_name}')
    try:
        dataset = load_dataset(options.data_name, data_files=data_files, cache_dir=options.cache)
    except FileNotFoundError:
        raise FileNotFoundError("""Cannot read the dataset. Check that given language(s)="+str(options.language)+" exist(s) in the dataset,
        or specify --data_type='csv' if you're trying to read a local file.""")
    return dataset

def read_dataset(options):

    if options.data_type=="huggingface":
        dataset = read_huggingface(options)
    elif options.data_type=="oscar":
        dataset = read_oscar(options)
    elif options.data_type=="csv":
        dataset = read_csv(options)
        
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
        # NA is changed to NaN unfortunately, correcting that:
        if d["labels"] == None and "NA" in options.labels:
            d["labels"]="NA"
        # if the labels are not as a list, correcting that:
        if type(d["labels"])!= list:
            d["labels"] = d["labels"].replace(" ",",").split(",")
        # only keeping labels we're interested in
        d["labels"] = [i for i in d["labels"] if i in options.labels]
        # removing punctuation
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

# testing...
if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    #print(options)
    dataset = read_dataset(options)
    print(dataset)
    dataset, tok=process_dataset(dataset,options)
    print(dataset)
