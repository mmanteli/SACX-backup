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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from train_multilabel import read_dataset, wrap_preprocess, binarize
import json
from tqdm import tqdm


labels1 = ['HI', 'ID', 'IN', 'IP', 'NA', 'OP']
int_bs = 16
CACHE = "/scratch/project_2002026/amanda/cache/"


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--base_model','--model_name', default="xlm-roberta-base",
                    help='Base model name for tokenizer')
    ap.add_argument('--trained_model', default=None, required=True,
                    help='Path to trained model')
    ap.add_argument('--data_name',type=str, metavar='HF-DATASETNAME', default="TurkuNLP/register_oscar",
                    help='Name of the dataset')
    ap.add_argument('--language',  type=json.loads,default=["en"], metavar='LIST-LIKE',
                    help='Language to be used from the dataset. Give as \'["en","zh"]\' ')
    ap.add_argument('--labels', type=json.loads, default=labels1, metavar='LIST-LIKE',
                    help='which labels to use, give as \'["IN","NA"]\' ')
    ap.add_argument('--split', metavar='FLOAT', type=float, default=0.8,
                    help='Set train/val data split ratio, e.g. 0.8 to train on 80 percent')
    ap.add_argument('--downsample', metavar="BOOL", type=bool,
                    default=True, help='downsample to 1/20th')
    ap.add_argument('--visualize', metavar="BOOL", type=bool,default=False,
                    help='If True, print HTML presentation. NOTE: this PRINTS, so redirect output using ">"!')
    ap.add_argument('--cache', default=CACHE, metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--int_batch_size', metavar='INT', type=int, default=int_bs,
                    help='Batch size for integrated gradients')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=123, help='Random seed for splitting data')
    ap.add_argument('--save_file', default="./explanations/exp", metavar='FILE',
                    help='Path to save file. "_language.tsv" added in the script.')
    return ap




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

def aggregate(inp,attrs,tokenizer):
    """detokenize and merge attributions"""
    detokenized=[]
    for l in inp.input_ids.cpu().tolist():
        detokenized.append(tokenizer.convert_ids_to_tokens(l))
    attrs=attrs.cpu().tolist()
    aggregated=[]
    for token_list,attr_list in zip(detokenized,attrs): #One text from the batch at a time!
        res=[]
        for token,a_val in zip(token_list,attr_list):
            if token == "<s>" or token == "</s>":  # special tokens
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



def explain(text,model,tokenizer,wrt_class="winner", int_bs=10, n_steps=50):
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
    for key in options.language:
        # all results saved here
        save_matrix = []
        print(f'Explaining {key}...')

        # loop over the dataset. Index "i" used here in case the dataset does not contain "id".
        for i in tqdm(range(len(dataset["validation_"+str(key)]))):
            # prepocess everything that is needed for saving and calculation
            txt = dataset["validation_"+str(key)]['text'][i]
            lbl = dataset["validation_"+str(key)]['labels'][i]
            try:
                id = dataset["validation_"+str(key)]['id'][i]
            except:
                id = 'document_'+str(i)
            # change label first to index format and then to class name => same happens to prediction later
            lbl = [options.id2label[i] for i,v in enumerate(lbl[0]) if v ==1]
            if len(lbl)==0:
                lbl=None    # change empty to None for later easy filtering
            if txt == None:
                txt = " "   # for empty sentences

            # do a prediction and explanation
            target, aggregated, probs = explain(txt, model, tokenizer, int_bs=options.int_batch_size)
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

        # save the results
        filename = options.save_file+"_"+str(options.seed)+"_"+key+'.tsv'
        pd.DataFrame(save_matrix, columns=["id", "label","pred","token","score","probs"]).to_csv(filename, sep="\t")
        print("Dataset "+ key +" succesfully saved")
        
    #return save_matrix
    

if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    
    tokenizer = AutoTokenizer.from_pretrained(options.base_model)
    model = torch.load(options.trained_model)
    model.to('cuda')
    print("Model loaded succesfully.")

    options.id2label = model.config.id2label

    # load the test data
    dataset = read_dataset(options)
    dataset = dataset.map(wrap_preprocess(options))
    dataset, mlb_classes = binarize(dataset, options)
    
    print("Dataset loaded. Ready for explainability.\n")

    # loop over languages
    explain_and_save_documents(dataset, model, tokenizer, options)
