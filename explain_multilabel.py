from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import transformers
from transformers import AutoTokenizer
import captum
import re
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from datasets import load_dataset
import pandas as pd
import csv
#from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from arguments import argparser
import sys
from data_preparation import read_dataset, wrap_preprocess, binarize
from sklearn.metrics import classification_report
import json
from tqdm import tqdm
from importlib import import_module
import spacy
import warnings

SPECIAL_TOKENS=["<s>", "</s>"]

# # Forward on the model -> data in, prediction out, nothing fancy really
def predict(model, inputs, int_bs=None, attention_mask=None):
    pred=model(inputs, attention_mask=attention_mask) # TODO: batch_size?
    return pred.logits #return the output of the classification layer

def blank_reference_input(tokenized_input, blank_token_id): #b_encoding is the output of HFace tokenizer
    """
    Makes a tuple of blank (input_ids, token_type_ids, attention_mask)
    right now position_ids, and attention_mask simply point to tokenized_input
    """

    blank_input_ids=tokenized_input.input_ids.clone().detach()
    blank_input_ids[tokenized_input.special_tokens_mask==0]=blank_token_id #blank out everything which is not special token
    return blank_input_ids, tokenized_input.attention_mask

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


#---------------------------------from predictions to words--------------------------------#

def aggregate(inp,attrs,tokenizer):
    """Detokenize and merge attributions. This works for languages that use white spaces between the words."""
    detokenized=[]
    for l in inp.input_ids.cpu().tolist():
        detokenized.append(tokenizer.convert_ids_to_tokens(l))
    attrs=attrs.cpu().tolist()
    aggregated=[]
    for token_list,attr_list in zip(detokenized,attrs): #One text from the batch at a time!
        res=[]
        for token,a_val in zip(token_list,attr_list):
            if token in SPECIAL_TOKENS:  # special tokens
                res.append((token,a_val))
            elif token.startswith("▁"):
                #This NOT is a continuation. A NEW word.
                res.append((token[1:],a_val))
                #print(res)
            else:  # we're continuing a word and need to choose the larger abs value of the two
                last_a_val = res[-1][1]
                #print("last val", last_a_val)
                if abs(a_val)<abs(last_a_val): #past value bigger
                    res[-1]=(res[-1][0]+token, last_a_val)
                else:  #new value bigger
                    res[-1]=(res[-1][0]+token, a_val)

        aggregated.append(res)
    return aggregated

ASSERTION_MSG = lambda p,t,sp: '''
    Given parsing of the sentence cannot be aligned.
    Checking these might help:
    - if string.isalpha() is applicable for your language
    - the variable "special_tokens" contains the special tokens of your model,
    - the two tokenizations are of the same sentence.
    ''' + f'\nThe sentences are \n {p} \n {t} \nand the special tokens are {sp}.'

def align(inp_text, inp, inp_scores, tokenizer,parser):
    """
    Function to align two different tokenizations of the same text.
    E.g. for Chinese, the model tokenizer might tokenize
    母语 (mother tongue) as "母" and "语".
    -> if you have defined a better tokenizer, give both tokenizations
    to this function and it will align them and the scores
    associated with the model tokenizer.
    All punctuation is removed for this tokenization.
    -> would be removed in the keyword extraction step anyway.
    """
    
    # preprocess both
    # start by transforming this to text:
    tokenized=[]
    scores = []
    for l,s in zip(inp.input_ids.cpu().tolist()[0], inp_scores.cpu().tolist()[0]):
        tl = tokenizer.convert_ids_to_tokens(l)
        if tl not in SPECIAL_TOKENS:
            tokenized.append(tl)
            scores.append(s)
    #tokenized=tokenized[0]  #flatten

    # this serves as a new text: TODO: check that this matches inp_text
    text = "".join(tokenized)
    #assert 
    parsed = [token.text for token in parser(text)]

    
    # drop things that are tokenized differently, i.e. spacy and roberta handle commas differently etc.
    parsed = [re.sub(r'[^\D\s]|[^\w\s]', '', p) for p in parsed if any(j.isalpha() for j in p)]
    to_be_dropped = [i for i in range(len(tokenized)) if any(j.isalpha() for j in tokenized[i])]
    tokenized = [re.sub(r'[^\D\s]|[^\w\s]', '',tokenized[i]) for i in to_be_dropped]
    scores = [scores[i] for i in to_be_dropped]

    #print("------------------text------------------------")
    #print(text)
    #print("-----------------parsed-----------------------")
    #print(parsed)
    #print("----------------tokenized---------------------")
    #print(tokenized)
    

    assert "".join(parsed)=="".join(tokenized), ASSERTION_MSG(parsed, tokenized, SPECIAL_TOKENS)
    
    # align
    # t_ind contains the current index we're at now in the tokenized (by model) sentence
    t_ind=0
    agg_scores = np.zeros(len(parsed))
    # for each "real tokenization" (here "parsed")
    for p_ind in range(len(parsed)):
        sub_scores = []
        for p in parsed[p_ind]:     # for each char in that tokenization
            if p == tokenized[t_ind][0]:        # if theres a match
                sub_scores.append(scores[t_ind])    # add the score for the character
                if len(tokenized[t_ind])>1:         # remove the character for next calculation
                    tokenized[t_ind] = tokenized[t_ind][1:]
                else:
                    tokenized[t_ind]="-"        # not necessary but helped debug
                    t_ind+=1                    # move to next index
            else:
                raise Exception("Alignment impossible for unforeseen reasons.") # despite assertion, something went wrong
        agg_scores[p_ind] = np.max(sub_scores)     # aggregate the scores TODO: method

    return [[(p,a) for p,a in zip(parsed,agg_scores.tolist())]]#[parsed], [agg_scores.tolist()]

#explain(key, txt, model, tokenizer, options, int_bs=options.int_batch_size)
def explain(lang, text,model,tokenizer, options, wrt_class="winner", int_bs=10, n_steps=50):
    # white space inbetween punctuation => for standard tokenisation
    text = re.sub('(?<! )(?=[:.,!?()])|(?<=[:.,!?()])(?! )', r' ', text) 
    # Tokenize and make the blank reference input
    inp = tokenizer(text,return_tensors="pt",return_special_tokens_mask=True,truncation=True).to(model.device)
    b_input_ids, b_attention_mask=blank_reference_input(inp, tokenizer.convert_tokens_to_ids("-"))


    def predict_f(inputs, attention_mask=None):
        return predict(model,inputs,attention_mask=attention_mask)

    # Here's where the magic happens
    lig = LayerIntegratedGradients(predict_f, model.roberta.embeddings)
    if wrt_class=="winner":
        # make a prediction
        prediction=predict(model,inp.input_ids, attention_mask=inp.attention_mask)
        # get the classification layer outputs
        logits = prediction.cpu().detach().numpy()[0]
        # calculate sigmoid for each
        sigm = 1.0/(1.0 + np.exp(- logits))
        # make the classification, threshold = 0.5
        target = np.array([pl > 0.5 for pl in sigm]).astype(int)
        # get the classifications' indices
        target = np.where(target == 1)
        # return None if no classification was done
        if len(target[0]) == 0: # escape early if no prediction
            return None, None, sigm

    else:
        target = wrt_class


    aggregated = []
    # loop over the targets => "[0]" to flatten the extra dimension, actually looping over all targets
    for tg in target[0]:
        attrs, delta= lig.attribute(inputs=(inp.input_ids,inp.attention_mask),
                                     baselines=(b_input_ids,b_attention_mask),
                                     return_convergence_delta=True,target=tuple([np.array(tg)]),internal_batch_size=int_bs,n_steps=n_steps)
        # append the calculated and normalized scores to aggregated
        attrs_sum = attrs.sum(dim=-1)
        attrs_sum = attrs_sum/torch.norm(attrs_sum)
        if options.parse_separately is not None and lang in options.parse_separately:
            print(f'Using different parser for {lang}.')
            aggregated_tg = align(text, inp, attrs_sum, tokenizer, options.parser)
        else:
            aggregated_tg=aggregate(inp,attrs_sum,tokenizer)
        aggregated.append(aggregated_tg)

    # these are wonky but will have dim numberofpredictions x 1
    return target,aggregated,sigm


def print_aggregated(target,aggregated,real_label):
    """"
    This requires one target and one agg vector at a time
    Shows agg scores as colors
    """
    print("<html><body>")
    for tg,inp_txt in zip(target,aggregated): #one input of the batch
        x=captum.attr.visualization.format_word_importances([t for t,a in inp_txt],[a for t,a in inp_txt])
        print(f"<b>prediction: {tg}, real label: {real_label}</b>")
        print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")

def print_scores(target, aggregated, idx):
    """"
    Prints doc_id, label, token and agg score.
    Used for testing.
    """
    for tg, ag in zip(target[0], aggregated):
        target = tg
        aggregated = ag
        for tok,a_val in aggregated[0]:
            if a_val > 0.05:
                print("document_"+str(idx),labels[target],str(tok),a_val,sep="\t")


def explain_and_save_documents(dataset, model, tokenizer, options):
    """
    loop over the data: for each, predict it, and for each predicted class, calculate 
    word attribution/effect it had on the classification. 
    Save each token as one line; if document 0 has been predicted as [2,3] but is really [2,4], 
    the output will look like this:
    DOC    label  pred  token score    probs
    doc_0  [2,4]  2     this  (score)  (sigm)
    doc_0  [2,4]  2     is    (score)  (sigm)
    doc_0  [2,4]  2     a     (score)  (sigm)
    doc_0  [2,4]  2     doc   (score)  (sigm)
    doc_0  [2,4]  3     this  (score)  (sigm)
    doc_0  [2,4]  3     is    (score)  (sigm)
    doc_0  [2,4]  3     a     (score)  (sigm)
    doc_0  [2,4]  3     doc   (score)  (sigm)
    The scores are calculate wrt predicted class, hence two sets of scores are needed.
    """
    reports = []
    for key in options.language:
        # all results saved here
        save_matrix = []
        print(f'Explaining {key}...')

        # collect
        errors = 0
        trues = []
        preds = []
        # loop over the dataset. Index "i" used here in case the dataset does not contain "id".
        for i in tqdm(range(len(dataset["validation_"+str(key)]))):
            # prepocess everything that is needed for saving and calculation
            txt = dataset["validation_"+str(key)]['text'][i]
            lbl_ = dataset["validation_"+str(key)]['labels'][i]
            try:
                id = dataset["validation_"+str(key)]['id'][i]
            except:
                id = 'document_'+str(i)
            # change label first to index format and then to class name => same happens to prediction later
            lbl = [options.id2label[i] for i,v in enumerate(lbl_[0]) if v ==1]
            if len(lbl)==0:
                lbl=None    # change empty to None for later easy filtering
            if txt == None:
                txt = " "   # for empty sentences

            # do a prediction and explanation
            try:
                target, aggregated, probs = explain(key, txt, model, tokenizer, options, int_bs=options.int_batch_size)
                if target != None:
                    # for all labels, tokens, and their agg scores: save a line in the document
                    for tg, ag in zip(target[0], aggregated):
                        target = tg
                        aggregated = ag
                        for tok,a_val in aggregated[0]:
                            line = [id, str(lbl), [options.id2label[target]], str(tok), a_val, probs.tolist()]
                            save_matrix.append(line)
                        # print visualisation in HTML format
                        if options.visualize:
                            print_aggregated([options.id2label[target]],aggregated,lbl)
                else:  #for no prediction, save None for target and score    TODO: Maybe still tokenize here?
                    for word in txt.split():
                        line = [id, str(lbl), "None", word, "None", probs.tolist()]
                        save_matrix.append(line)
                trues.append(lbl_[0])
                p = np.zeros(len(options.label2id), dtype=int)
                if target != None:
                    p[target] = 1
                preds.append(p)
            except Exception as e:
                file1 = open("errors.err", "a")
                file1.write(str(e))
                file1.write("\n\n")
                errors+=1
                continue
            
        print(f'ENCOUNTERED {errors} error(s)!')
        # save the results
        filename = options.save_file+"_"+str(options.seed)+"_"+key+'.tsv'
        pd.DataFrame(save_matrix, columns=["id", "label","pred","token","score","probs"]).to_csv(filename, sep="\t")
        print("Dataset "+ key +" succesfully saved")
        reports.append(classification_report(trues, preds, target_names = options.label2id.keys()))
        
    return save_matrix, reports
    


if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    if options.parse_separately is not None and options.parser_model is not None:
        options.parser = spacy.load(options.parser_model)
        print(f'{options.parser_model} loaded from spacy.')
    print(options)

    tokenizer = AutoTokenizer.from_pretrained(options.model_name)
    try:
        model = torch.load(options.save_model)
    except:
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(options.save_model)
        except:
            print("Cannot load model using either torch or transformers.AutoModel")
    model.to('cuda')
    print("Model loaded succesfully.")


    # load the test data, and define label2id/id2label based on the data
    dataset = read_dataset(options)
    dataset = dataset.map(wrap_preprocess(options))
    dataset, mlb_classes = binarize(dataset, options)
    options.label2id = dict(zip(mlb_classes, [i for i in range(0, len(mlb_classes))]))
    options.id2label = {v:k for k,v in options.label2id.items()}
    

    if model.config.id2label != options.id2label:
        warnings.warn(f'Model id2label dictionary and the one extracted from data do not match. \n \
        Model id2label: {model.config.id2label},\n \
        Data id2label: {options.id2label},\n \
        Changing model config to follow data. ')
        model.config.id2label = options.id2label
        model.config.label2id = {v:k for k,v in options.id2label.items()}
    
    print("Dataset loaded. Ready for explainability.\n")

    # loop over languages
    m, reports = explain_and_save_documents(dataset, model, tokenizer, options)
    for r in reports:
        print(r)
