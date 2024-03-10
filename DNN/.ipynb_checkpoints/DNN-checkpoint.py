import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation_function='sigmoid', learning_rate=0.01):
        # 初始化权重矩阵、层数、学习率
        # 例如：layers=[1, 10, 1]，表示输入层 1 个结点，隐藏层 10 个结点，输出层 1 个结点
        self.W = []
        self.layers = layers
        self.activation_function = activation_function
        self.learning_rate = learning_rate

		# 随机初始化权重矩阵，如果三层网络，则有两个权重矩阵；
        # 在初始化的时候，对每一层的结点数加1，用于初始化训练偏置的权重；
        # 由于输出层不需要增加结点，因此最后一个权重矩阵需要单独初始化；
        #  y = x * w
        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # 初始化最后一个权重矩阵
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
        print(len(self.W))

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)

    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)

    def fit(self, X, y, epochs=10000):
        print(X)
        X = np.c_[X, np.ones(X.shape[0])]
        print(X)
        print()
        print([np.atleast_2d(X)])
        # for i in range(epochs):
        #     output = self.forward(X)
        #     dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1 = self.backward(X, y, output)
        #     self.update_parameters(dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1)
        #     if i % 1000 == 0:
        #         print(f'Epoch {i}: Loss {np.mean((output - y)**2)}')

layers = [1, 10, 10, 10, 10, 1]

X = np.linspace(-10, 10, 4).reshape(-1, 1)
y = X**2 + 2*X - 3

nn = NeuralNetwork(layers=layers, activation_function='relu', learning_rate=0.001)

nn.fit(X.T, y.T)