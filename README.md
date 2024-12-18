# SACX-backup

Further development for SACX keyword extraction pipeline. See [TurkuNLP class-explainer](https://github.com/TurkuNLP/class-explainer) for the implementation used in the original paper.

## What it does?

- Trains multilabel classification models and uses the Integrated Gradients method to explain the results of the classifier.
  - i.e. IG method scores words based on their relevance in the classification
  - We're trying to see which words consistently score high -> keywords
- Aggregates the results over all trained models and produces lists of keywords for each class in the classification.

## How to run?

### Requirements:

On LUMI, module ``PyTorch`` contains most of requirements:

```
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
```

However, ``captum`` and ``spacy`` are needed for explanation scores and word-border parsing in some languages (e.g. Chinese, Thai). For this virtual environment is usually created but as they are not recommended on LUMI (and mess up GPU partitions specifically), create a ``PYTHONUSERBASE`` instead:

```
# Move to a location you want to save the userbase
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
mkdir PYTHONUSERBASE  # or any name you want
cd PYTHOUSERBASE
PYTHONUSERBASE=$(pwd)   # saving to environment variable
export PYTHONUSERBASE
python -m pip install --user captum==0.7.0
python -m pip install --user spacy==3.7.4
python -m spacy install --user <parsing models for the languages you wish to analyse, e.g. zh_web_core_md, see Multiligual below>
```
**Note** that calling ``python`` usually does not work on LUMI, but works now because we downloaded ``PyTorch``. 

Now anywhere you would normally ``source venv/bin/activate`` instead 
```export PYTHONPATH=<path to PYTHONUSERBASE>/PYTHONUSERBASE/lib/python3.10/site-packages:$PYTHONPATH```

If something else is needed to install, repeat the steps above (except making the PYTHONUSERBASE).

### From scratch:

- run ``sl-train-explain`` (calls ``train_and_explain.py``) with different seed values:
  - with Slurm: run ``sbatch sl-train-explain.bash <insert seed value here>`` with 10+ different seed values.
  - without Slurm: run ``python train_and_explain.py --seed=<insert seed value here>`` with 10+ different seed values.
  - Create folder "explanations" beforehand, unless you define the parameter ``--explanations=<new path>`` to point somewhere else.
  - you can modify these parameters on the command line/slurm script or in the file ``arguments.py``.
- run ``sl-kws.sh`` (calls ``kws.py``) with the results produced by training-explaining.
- If needed, corpus information can be used. In that case, before ``kws.py``, run ``count_class_words.py`` which calculates term and document frequencies.
  - give the result as a parameter --corpus_information to ``kws.py`` and a limit for term frequency as ``--min_word_freq``

### With already-trained models

- run ``sl-explain.sh`` with all the different models you have as parameter ``--trained_model`` (note that basename is still needed for tokenizer!)
- run ``sl-kws.sh`` similarly to above

## Multilingual?

- you can give multiple languages as --language='["en", "fr", "zh"]' etc.
- trains with all languages, explains them separately.
- for laguages that do not use white space to separate words, you can use a spacy-parser by defining for example ``--parse_separately="zh"`` and ``--parser_model="zh_core_web_md"``
  - in this case, feed the languages in separately, so that only one parser is loaded at a time. I.e. if there is only one language that needs separate parsing, this can be done:

```
python train_and_explain.py --language='["en", "fr", "jp"]' <other params> --parse_separately="jp" --parser_model="jp_core_news_md"
```

 - ... but if you have, say "zh" and "jp", then they need to be explained separately:

```
python train_multilabel.py --language='["jp", "zh"]' --save_model=<path to model>
python explain_multilabel.py --trained_model=<path to model> --language='["jp"]' --parse_separately="jp" --parser_model="jp_core_news_md"
python explain_multilabel.py --trained_model=<path to model> --language='["zh"]' --parse_separately="zh" --parser_model="zh_core_web_md"
```

