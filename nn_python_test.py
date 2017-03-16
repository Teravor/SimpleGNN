import numpy as np
import matplotlib.pyplot as plt
'''
Very simple NN base on sigmoids.
'''

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def dsigmoid(y):
	return y*(1.0 - y)

class Layer:
	def __init__(self, sI, sO):
		self.weights = np.random.randn(sO, sI)
		self.bias = np.random.randn(sO)
		self.output = np.zeros(sO)

	def feedforward(self, x):
		#import pdb;pdb.set_trace()
		self.output = sigmoid(np.dot(self.weights, x) + self.bias)

	def update_parameters(self, a_j, delta, rate):
		result = np.outer(delta, a_j)
		if(result.shape != self.weights.shape):
			import pdb;pdb.set_trace()
		self.weights += rate*np.outer(delta, a_j)
		self.bias += rate*delta

	def backpropagate(self,a_j,delta):
		#import pdb;pdb.set_trace()
		delta = np.dot(self.weights.T,delta)*dsigmoid(a_j)
		#import pdb;pdb.set_trace()
		return delta

class Network:
	def __init__(self, sI, sO, hidden_layers = []):
		self.layers = []
		layers_ = [sI] + hidden_layers + [sO]
		for i in range(0, len(hidden_layers)+1):
			self.layers.append(Layer(layers_[i], layers_[i+1]))
		self.output = None

	def compute(self, x):
		if(np.isscalar(x)):
			x = np.array([x])
		prev_output = x
		for layer in self.layers:
			layer.feedforward(prev_output)
			prev_output = layer.output
		self.output = self.layers[-1].output
		return self.output

	def train(self,arr_x,arr_y, rate):
		for x,y in zip(arr_x, arr_y):
			if(np.isscalar(x)):
				x = np.array([x])
			self.compute(x)
			self.backpropagate(x, y, rate)


	def backpropagate(self, x, y, rate):
		#Calculate first error
		delta = dsigmoid(self.output)*(y - self.output)
		for layerj, layerk in reversed(zip(self.layers[:-1], self.layers[1:])):
			layerk.update_parameters(layerj.output, delta, rate)
			delta = layerk.backpropagate(layerj.output, delta)
		self.layers[0].update_parameters(x, delta, rate)

	def print_layer_outputs(self):
		for i,layer in enumerate(self.layers):
			print("Layer {0}, output shape {1}".format(i, layer.output.shape))
			print("Parameters shapes {0}, {1}".format(layer.weights.shape, layer.bias.shape))

def calc_err(n, x, y):
	nn_pred = []
	for xi in x:
		nn_pred.append(n.compute(xi)[0])
	#import pdb;pdb.set_trace()
	err = np.sum((y - nn_pred)**2)
	return err

def test1():
	num = 1
	n = Network(1,1, [20])
	result = n.compute(1)
	print(result)



if __name__ == "__main__":
	num = 100
	x = np.linspace(0, 3, num)
	y = np.sin(x)
	epochs = 2000
	n = Network(1, 1, [20,20])
	epoch_errors = []
	rate = 0.05

	for i in range(epochs):
		if(i % 25 == 0	):
			print("{0} Epochs done".format(i))
		n.train(x,y,rate)
		epoch_errors.append((calc_err(n, x, y)))
	nn_pred = []
	for xi in x:
		nn_pred.append(n.compute(xi)[0])
	
	plt.figure()
	plt.title("Error over epochs")
	plt.plot(epoch_errors)
	
	
	plt.figure()
	plt.title("Neural network and true function")
	plt.plot(x, y, label="Sin(x)")
	plt.plot(x, nn_pred, label="NN")
	plt.legend()

	plt.show()