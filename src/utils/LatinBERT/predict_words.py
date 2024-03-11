"""
	This is a modified version of the predict_words code found in the latin-bert github: https://github.com/dbamman/latin-bert/blob/master/case_studies/infilling/scripts/predict_word.py .

"""

import argparse
import copy, re
import sys
from transformers import BertModel, BertForMaskedLM, BertPreTrainedModel
from tensor2tensor.data_generators import text_encoder
import torch
import numpy as np
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from .LatinTok import LatinTokenizer
from torch import nn
import random

random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def proc(tokenids, wp_tokenizer, model):

	mask_id=tokenids.index(wp_tokenizer.vocab["[MASK]"])
	torch_tokenids=torch.LongTensor(tokenids).unsqueeze(0)
	torch_tokenids=torch_tokenids.to(device)
	
	with torch.no_grad():
		preds = model(torch_tokenids)
		preds=preds[0]
		sortedVals=torch.argsort(preds[0][mask_id], descending=True)
		for k, p in enumerate(sortedVals[:1]):
			predicted_index=p.item()
			probs=nn.Softmax(dim=0)(preds[0][mask_id])
			return wp_tokenizer.reverseVocab[predicted_index]
		
def infilling(wp_tokenizer, text_before_pred, text_after_lacuna, model):
	"""
		This takes in a tokenizer, the text before the masked text, the text after it, and the 
		model to perform infilling.
	"""
	tokens=[]
	tokens.extend(wp_tokenizer.tokenize(text_before_pred))
	position=len(tokens) + 1
	tokens.append("[MASK]")
	tokens.extend(wp_tokenizer.tokenize(text_after_lacuna))

	tokens.insert(0,"[CLS]")
	tokens.append("[SEP]")

	tokenids=wp_tokenizer.convert_tokens_to_ids(tokens)	

	mask_id=tokenids.index(wp_tokenizer.vocab["[MASK]"])

	torch_tokenids=torch.LongTensor(tokenids).unsqueeze(0)
	torch_tokenids=torch_tokenids.to(device)
	total_text = ""
	with torch.no_grad():
		preds = model(torch_tokenids)
		preds = preds[0]
		
		sortedVals=torch.argsort(preds[0][mask_id], descending=True)
		p = sortedVals[0]
		predicted_index=p.item()
		probs=nn.Softmax(dim=0)(preds[0][mask_id])


		suffix=""
		if not wp_tokenizer.reverseVocab[predicted_index].endswith("_"):
			uptokens=copy.deepcopy(tokenids)
			uptokens.insert(position, predicted_index)
			suffix=proc(uptokens, wp_tokenizer, model)
		
		predicted_word="%s%s" % (wp_tokenizer.reverseVocab[predicted_index], suffix)
		predicted_word=re.sub("_$", "", predicted_word).lower()
		total_text = text_before_pred + " " + predicted_word + " " + text_after_lacuna
	
	return total_text

		
def	predict(wp_tokenizer, text_before_pred,	model,	context_size:int=256,	k:int	=	20,	sampleK:bool	=	False):
	"""
		The predict function follows the same methodology as infilling, except the there is only text before the 
		prediction and then, if sampling is allowed, it samples from the top k predicted words.
	"""
	tokens=[]
	tokens.extend(wp_tokenizer.tokenize(text_before_pred))
	position=len(tokens)	+	1
	tokens.append("[MASK]")
	pads = context_size-len(tokens)-2
	tokens+=["[PAD]"]*pads
	
	tokens.insert(0,"[CLS]")
	tokens.append("[SEP]")
	

	tokenids=wp_tokenizer.convert_tokens_to_ids(tokens)	

	mask_id=tokenids.index(wp_tokenizer.vocab["[MASK]"])

	torch_tokenids=torch.LongTensor(tokenids).unsqueeze(0)
	torch_tokenids=torch_tokenids.to(device)
	total_text	= ""
	with torch.no_grad():
		preds =	model(torch_tokenids)
		preds =	preds[0]
		
		sortedVals=torch.argsort(preds[0][mask_id],	descending=True)
		idx	= 0
		if	sampleK:
			idx	= random.randint(0,k)
			
		p = sortedVals[idx]
		predicted_index=p.item()
		probs=nn.Softmax(dim=0)(preds[0][mask_id])


		suffix=""
		if	not	wp_tokenizer.reverseVocab[predicted_index].endswith("_"):
			uptokens=copy.deepcopy(tokenids)
			uptokens.insert(position,	predicted_index)
			suffix=proc(uptokens,	wp_tokenizer,	model)
		
		predicted_word="%s%s"	%	(wp_tokenizer.reverseVocab[predicted_index],	suffix)
		predicted_word=re.sub("_$",	"",	predicted_word).lower()
		total_text = text_before_pred + " " + predicted_word
	
	return	total_text

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bertPath',	help='path	to	pre-trained	BERT',	required=True)
	parser.add_argument('-t', '--tokenizerPath', help='path	to	Latin	WordPiece	tokenizer',	required=True)
	
	args = vars(parser.parse_args())

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]			
	
	encoder	= text_encoder.SubwordTextEncoder(tokenizerPath)
	wp_tokenizer = LatinTokenizer(encoder,lowercase=True)

	model = BertForMaskedLM.from_pretrained(bertPath)
	model.to(device)
	predict(encoder,"arma virum canoque",model)
