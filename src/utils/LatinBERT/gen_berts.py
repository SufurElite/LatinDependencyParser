"""
	This file is effectively the same as the LatinBERT one, the imports are different 
	and some modifications had been tried.  The purpose of it is mostly to have the Class architecture for
	LatinBERT.

	There was also an error in the LatinBERT file due to deprecation. If you see the 'todo' at the bottom, 
	I realised that the transformers library had an update that was required for this file to work.

"""
import argparse, sys
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from .LatinTok import LatinTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LatinBERT():

	def __init__(self, tokenizerPath=None, bertPath=None):
		encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
		self.wp_tokenizer = LatinTokenizer(encoder)
		self.model = BertLatin(bertPath=bertPath)
		self.model.to(device)

	def get_batches(self, sentences, max_batch, tokenizer):

			maxLen=0
			for sentence in sentences:
				length=0
				for word in sentence:
					toks=tokenizer.tokenize(word)
					length+=len(toks)

				if length> maxLen:
					maxLen=length

			all_data=[]
			all_masks=[]
			all_labels=[]
			all_transforms=[]

			for sentence in sentences:
				tok_ids=[]
				input_mask=[]
				labels=[]
				transform=[]

				all_toks=[]
				n=0
				for idx, word in enumerate(sentence):
					toks=tokenizer.tokenize(word)
					all_toks.append(toks)
					n+=len(toks)

				cur=0
				for idx, word in enumerate(sentence):
					toks=all_toks[idx]
					ind=list(np.zeros(n))
					for j in range(cur,cur+len(toks)):
						ind[j]=1./len(toks)
					cur+=len(toks)
					transform.append(ind)

					tok_ids.extend(tokenizer.convert_tokens_to_ids(toks))

					input_mask.extend(np.ones(len(toks)))
					labels.append(1)

				all_data.append(tok_ids)
				all_masks.append(input_mask)
				all_labels.append(labels)
				all_transforms.append(transform)

			lengths = np.array([len(l) for l in all_data])

			# Note sequence must be ordered from shortest to longest so current_batch will work
			ordering = np.argsort(lengths)
			
			ordered_data = [None for i in range(len(all_data))]
			ordered_masks = [None for i in range(len(all_data))]
			ordered_labels = [None for i in range(len(all_data))]
			ordered_transforms = [None for i in range(len(all_data))]
			

			for i, ind in enumerate(ordering):
				ordered_data[i] = all_data[ind]
				ordered_masks[i] = all_masks[ind]
				ordered_labels[i] = all_labels[ind]
				ordered_transforms[i] = all_transforms[ind]

			batched_data=[]
			batched_mask=[]
			batched_labels=[]
			batched_transforms=[]

			i=0
			current_batch=max_batch

			while i < len(ordered_data):

				batch_data=ordered_data[i:i+current_batch]
				batch_mask=ordered_masks[i:i+current_batch]
				batch_labels=ordered_labels[i:i+current_batch]
				batch_transforms=ordered_transforms[i:i+current_batch]

				max_len = max([len(sent) for sent in batch_data])
				max_label = max([len(label) for label in batch_labels])

				for j in range(len(batch_data)):
					
					blen=len(batch_data[j])
					blab=len(batch_labels[j])

					for k in range(blen, max_len):
						batch_data[j].append(0)
						batch_mask[j].append(0)
						for z in range(len(batch_transforms[j])):
							batch_transforms[j][z].append(0)

					for k in range(blab, max_label):
						batch_labels[j].append(-100)

					for k in range(len(batch_transforms[j]), max_label):
						batch_transforms[j].append(np.zeros(max_len))

				batched_data.append(torch.LongTensor(batch_data))
				batched_mask.append(torch.FloatTensor(batch_mask))
				batched_labels.append(torch.LongTensor(batch_labels))
				batched_transforms.append(torch.FloatTensor(batch_transforms))

				bsize=torch.FloatTensor(batch_transforms).shape
				
				i+=current_batch

				# adjust batch size; sentences are ordered from shortest to longest so decrease as they get longer
				if max_len > 100:
					current_batch=12
				if max_len > 200:
					current_batch=6

			return batched_data, batched_mask, batched_transforms, ordering

	def get_bert_docs(self, raw_sents, labels, sent_tokenizer,word_tokenizer):
		sents_label=convert_to_toks(raw_sents,labels,sent_tokenizer,word_tokenizer)
		sents = [s[0] for s in sents_label]
		author= [s[1] for s in sents_label]

		batch_size=32
		batched_data, batched_mask, batched_transforms, ordering=self.get_batches(sents, batch_size, self.wp_tokenizer)
		ordered_preds=[]
		for b in range(len(batched_data)):
			size=batched_transforms[b].shape
			b_size=size[0]
			berts=self.model.forward(batched_data[b], attention_mask=batched_mask[b], transforms=batched_transforms[b])
			berts=berts.detach()
			berts=berts.cpu()
			for row in range(b_size):
				ordered_preds.append([np.array(r) for r in berts[row]])

		preds_in_order = [None for i in range(len(sents))]


		for i, ind in enumerate(ordering):
			preds_in_order[ind] = ordered_preds[i]


		bert_docs=[]

		for idx, sentence in enumerate(sents):
			bert_doc=[]

			bert_doc.append(preds_in_order[idx][0])

			for t_idx in range(1, len(sentence)-1):
				if t_idx==200-2: break
				token=sentence[t_idx]
				
				pred=preds_in_order[idx][t_idx]
				bert_doc.append(pred)

			bert_doc.append(preds_in_order[idx][len(sentence)-1])
			tmp = np.array(bert_doc)
			bert_docs.append(np.mean(tmp, axis=0))

		return bert_docs, author

	def get_berts(self, raw_sents, labels, sent_tokenizer,word_tokenizer):
		sents_label=convert_to_toks(raw_sents,labels,sent_tokenizer,word_tokenizer)
		sents = [s[0] for s in sents_label]
		author= [s[1] for s in sents_label]

		batch_size=32
		batched_data, batched_mask, batched_transforms, ordering=self.get_batches(sents, batch_size, self.wp_tokenizer)
		ordered_preds=[]
		for b in range(len(batched_data)):
			size=batched_transforms[b].shape
			b_size=size[0]
			berts=self.model.forward(batched_data[b], attention_mask=batched_mask[b], transforms=batched_transforms[b])
			berts=berts.detach()
			berts=berts.cpu()
			for row in range(b_size):
				ordered_preds.append([np.array(r) for r in berts[row]])

		preds_in_order = [None for i in range(len(sents))]


		for i, ind in enumerate(ordering):
			preds_in_order[ind] = ordered_preds[i]


		bert_sents=[]

		for idx, sentence in enumerate(sents):
			bert_sent=[]

			bert_sent.append(("[CLS]", preds_in_order[idx][0] ))

			for t_idx in range(1, len(sentence)-1):
				token=sentence[t_idx]
				
				pred=preds_in_order[idx][t_idx]
				bert_sent.append((token, pred ))

			bert_sent.append(("[SEP]", preds_in_order[idx][len(sentence)-1] ))

			bert_sents.append(bert_sent)

		return bert_sents, author



def convert_to_toks(input_sents,sents_label,sent_tokenizer,word_tokenizer):

	all_sents=[]
	leave_loop = False
	cur_count = 0
	last_auth = sents_label[0]
	for i in range(len(input_sents)):
		"""if i!=0 and last_auth!=sents_label[i]:
				cur_count= 0
				last_auth = sents_label[i-1]
		if cur_count == max_sent and last_auth==sents_label[i]: 
			continue
		else:
			last_auth = sents_label[i]
			cur_count = 0
		cur_count+=1"""
		text=input_sents[i].lower()

		sents=sent_tokenizer.tokenize(text)
		for j in range(len(sents)):
				
			tokens=word_tokenizer.tokenize(sents[j])
			filt_toks=[]
			filt_toks.append("[CLS]")
			for tok in tokens:
				if tok != "":
					filt_toks.append(tok)
			filt_toks.append("[SEP]")

			all_sents.append((filt_toks,sents_label[i]))

	return all_sents




class BertLatin(nn.Module):

	def __init__(self, bertPath=None):
		super(BertLatin, self).__init__()
		# had to update this for the new hugging face migration
		# TODO: make a PR updating LatinBERT later
		self.bert = BertModel.from_pretrained(bertPath, return_dict=False)
		self.bert.eval()
		
	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None):

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)
		sequence_outputs, pooled_outputs = self.bert.forward(input_ids, token_type_ids=None, attention_mask=attention_mask)

		all_layers=sequence_outputs
		out=torch.matmul(transforms,all_layers)
		return out