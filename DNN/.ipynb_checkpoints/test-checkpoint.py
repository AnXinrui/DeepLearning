import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, activation_function='sigmoid', learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W1 = np.random.randn(self.hidden_dim1, self.input_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((self.hidden_dim1, 1))
        self.W2 = np.random.randn(self.hidden_dim2, self.hidden_dim1) * np.sqrt(2.0 / self.hidden_dim1)
        self.b2 = np.zeros((self.hidden_dim2, 1))
        self.W3 = np.random.randn(self.hidden_dim3, self.hidden_dim2) * np.sqrt(2.0 / self.hidden_dim2)
        self.b3 = np.zeros((self.hidden_dim3, 1))
        self.W4 = np.random.randn(self.hidden_dim4, self.hidden_dim3) * np.sqrt(2.0 / self.hidden_dim3)
        self.b4 = np.zeros((self.hidden_dim4, 1))
        self.W5 = np.random.randn(self.output_dim, self.hidden_dim4) * np.sqrt(2.0 / self.hidden_dim4)
        self.b5 = np.zeros((self.output_dim, 1))

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

    def forward(self, X):
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.W3, self.a2) + self.b3
        self.a3 = self.activation(self.z3)
        self.z4 = np.dot(self.W4, self.a3) + self.b4
        self.a4 = self.activation(self.z4)
        self.z5 = np.dot(self.W5, self.a4) + self.b5
        self.a5 = self.z5  # 由于没有激活函数，直接作为输出
        return self.a5

    def backward(self, X, y, output):
        m = X.shape[1]
        
        # 计算输出层的误差
        delta5 = (output - y)

        # 计算第四层的误差
        delta4 = np.dot(self.W5.T, delta5) * self.activation_derivative(self.z4)

        # 计算第三层的误差
        delta3 = np.dot(self.W4.T, delta4) * self.activation_derivative(self.z3)

        # 计算第二层的误差
        delta2 = np.dot(self.W3.T, delta3) * self.activation_derivative(self.z2)

        # 计算第一层的误差
        delta1 = np.dot(self.W2.T, delta2) * self.activation_derivative(self.z1)

        # 计算权重和偏置的梯度
        dW5 = (1 / m) * np.dot(delta5, self.a4.T)
        db5 = (1 / m) * np.sum(delta5, axis=1, keepdims=True)
        dW4 = (1 / m) * np.dot(delta4, self.a3.T)
        db4 = (1 / m) * np.sum(delta4, axis=1, keepdims=True)
        dW3 = (1 / m) * np.dot(delta3, self.a2.T)
        db3 = (1 / m) * np.sum(delta3, axis=1, keepdims=True)
        dW2 = (1 / m) * np.dot(delta2, self.a1.T)
        db2 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)
        dW1 = (1 / m) * np.dot(delta1, X.T)
        db1 = (1 / m) * np.sum(delta1, axis=1, keepdims=True)

        return dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1

    def update_parameters(self, dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1):
        self.W5 -= self.learning_rate * dW5
        self.b5 -= self.learning_rate * db5
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y, epochs=100000):
        for i in range(epochs):
            output = self.forward(X)
            dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1 = self.backward(X, y, output)
            self.update_parameters(dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1)
            if i % 1000 == 0:
                print(f'Epoch {i}: Loss {np.mean((output - y)**2)}')

# 生成样本数据
X = np.linspace(-10, 10, 200).reshape(-1, 1)
y = X**2 + 2

#  input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, activation_function='sigmoid'
nn = NeuralNetwork(input_dim=1, hidden_dim1=10, hidden_dim2=10, hidden_dim3=10, hidden_dim4 = 10, output_dim=1, activation_function='relu', learning_rate=0.001)
nn.fit(X.T, y.T)

# 可视化结果
plt.scatter(X, y)
plt.plot(X, nn.forward(X.T).flatten(), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Neural Network Regression')
plt.show()
