from model import BiRNN
#from data import Word
import numpy
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms 
import matplotlib.pyplot as plt
from torch.autograd import Variable

input_size = 28 
hidden_size = 128 
batch_size = 100
seq_len = 28
num_layers = 2
num_classes = 10
num_epoch = 2

train_dataset = torchvision.datasets.MNIST(root='./data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/', train=False, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#rnn = RNN(input_size, hidden_size, batch_size, seq_len, num_layers, num_classes)
rnn = BiRNN(input_size, hidden_size, batch_size, seq_len, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.03)

def show_batch(batch):
	im = torchvision.utils.make_grid(batch)
	plt.imshow(numpy.transpose(im.numpy(), (1,2,0)))

def train():
	for epoch in range(num_epoch):
		for i, (images, labels) in enumerate(train_loader):
			# (batch_size, seq_len, input_size)
			#images = Variable(words.view(-1, seq_len, input_size)) 
			images = Variable(images.view(-1, seq_len, input_size))
			labels = Variable(labels)

			optimizer.zero_grad()
			outputs = rnn(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print per batch
			if (i+1) % 100 == 0: 
				# Step : [train_dx, batch_total]. e.g) #loop per 1 epoch
				print('Epoch [%d/%d], Step [%d/%d], Loss: %.3f' % (epoch+1, num_epoch, i+1, len(train_dataset)//batch_size, loss.data[0]))

def test():
	correct = 0
	total = 0
	for images, labels in test_loader:
		# (batch_size, seq_len, input_size)
		images = Variable(images.view(-1, seq_len, input_size))
		outputs = rnn(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum() 

		print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 

if __name__ == '__main__':
	print("[-]Train starts...")
	train()
	print("[+]Train ends...")
	print("[-]Test start...")
	test()
	print("[+]Test ends...")
	print("[-]Save the model")
	torch.save(rnn.state_dict(), 'rnn.pkl')
