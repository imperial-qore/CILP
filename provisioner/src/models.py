import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from .dlutils import *
from .constants import *
torch.manual_seed(1)

## Simple Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.002
		self.n_feats = feats
		self.n_window = 5 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats * self.n_window, self.n_feats), nn.Sigmoid())

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return self.fcn(g.view(-1))

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_window = 5
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.fcn = nn.Sequential(nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out = self.fcn(out.view(-1))
		return out

## Simple Transformer Model
class Transformer(nn.Module):
	def __init__(self, feats):
		super(Transformer, self).__init__()
		self.name = 'Transformer'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

## CILP Model
class CILP(nn.Module):
	def __init__(self, feats):
		super(CILP, self).__init__()
		self.name = 'CILP'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()
		self.likelihood = nn.Sequential(nn.Linear(self.n + 1, 1), nn.Sigmoid())

	def predwindow(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return memory, x

	def forward(self, src, tgt, hw):
		memory, x = self.predwindow(src, tgt)
		score = self.likelihood(torch.cat((hw, memory.view(-1))))
		return x, score