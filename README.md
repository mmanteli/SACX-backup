# SACX-backup
Backup for SACX keyword extraction pipeline

What it does?

- Runs multiple multilabel classification models and uses the Integrated Gradients method to explain the results of the classifier.
  - i.e. IG method scores words based on their relevance in the classification
  - We're trying to see which words consistently score high -> keywords
- Aggregates the results over all trained models and produces lists of keywords for each class in the classification.

How to run?

- run ``train_and_explain.py`` with different seed values:
  - with Slurm: run ``sbatch sl-train-explain.bash <insert seed value here>`` with 10+ different seed values.
  - without Slurm: run ``python train_and_explain.py --seed=<insert seed value here>`` with 10+ different seed values.
  - Create folder "explanations" beforehand, unless you define the parameter ``--explanations=<new path>`` to point somewhere else.
  - you can modify these parameters on the command line/slurm script or in the file ``arguments.py``.
- run ``kws.py`` with the results produced by training-explaining.
- If needed, corpus information can be used. In that case, before ``kws.py``, run ``count_class_words.py`` which calculates term and document frequencies.
  - give the result as a parameter --corpus_information to ``kws.py`` and a limit for term frequency as --min_word_freq
 
Multilingual?

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

