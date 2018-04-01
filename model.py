import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, seq_len, num_layers, num_classes):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.num_layers = num_layers

		self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# shape of x : (batch_size, seq_len, input_size)
		# shape of hidden : (num_layers, batch_size, hidden_size)
		h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
		c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

		out, hidden = self.rnn(x, (h0,c0))
		out = self.fc(out[:,-1,:])
		return out

class BIRNN(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, seq_len, num_layers, num_classes):
		super(BiRNN, self).__init__()
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.num_layers = num_layers

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
		# 2 for bidirection
		self.fc = nn.Linear(hidden_size*2, num_classes)

	def forward(self, x):
		h0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size))
		c0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size))		

		out, hidden = self.lstm(x, (h0,c0))
		# out : (batch_size, seq_len, hidden_size*2) / -1 : last_seq
		out = self.fc(out[:, -1, :]) 
		return out
