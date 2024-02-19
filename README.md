# SACX-backup
Backup for SACX keyword extraction pipeline

What it does?

- Runs multiple multilabel classification models and uses the Integrated Gradients method to explain the results of the classifier.
  - i.e. IG method scores words based on their relevance in the classification
  - We're trying to see which words consistently score high -> keywords
- Aggregates the results over all models and produces lists of keywords for each class.

How to run?

- run ``sl-train-explain.sh`` (Slurm) with different seed values. 10+ different runs recommended.
- run ``kws.py`` with the results produced by training-explaining.
- If needed, corpus information can be used. In that case, before ``kws.py``, run ``count_class_words.py`` which calculates term and document frequencies.
  - give the result as a parameter --corpus_information to ``kws.py`` and a limit for term frequency as --min_word_freq
 
Multilingual?

- you can give multiple languages as --language='["en", "fr", "zh"]' etc.
- trains with all languages, explains them separately.
