import numpy as np
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from torch import nn

class LatinTokenizer():
	def __init__(self, encoder, lowercase:bool = False):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder
		self.word_tokenizer = WordTokenizer()
		self.is_lower = lowercase
	
		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4

		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text):

		tokens=self.word_tokenizer.tokenize(text)

		wp_tokens=[]
		for token in tokens:
			
			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
			else:
				if self.is_lower:
					token=token.lower()
				
				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])

		return wp_tokens