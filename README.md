# 建立一个浅层神经网络

## 一、导入包

首先导入需要的所有包。

- numpy
- matplotlib
- sklearn
- textCases_v2  
- public_tests
- planar_utils

## 二、数据集

首先处理数据,将“flower” 二分类数据集加载到变量 `X` 和 `Y`中。

```python
X, Y = load_planar_dataset() 
```

可视化数据

```python
plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0,:].shape), s=40, cmap=plt.cm.Spectral)
```

![img](https://cdn.kesci.com/rt_upload/C7BA05618F6B4B609BF103B8ABD57E10/q17in84kex.png)

现在有：
  \- 包含特征（x1，x2）的numpy数组（矩阵）X
  \- 包含标签（红色：0，蓝色：1）的numpy数组（向量）Y。

目标是将颜色一样的点集合分类

如图

![img](https://cdn.kesci.com/rt_upload/CB068684F93C4A2A8AE816EB492CDCBE/q17hj9pr80.png)

## 三、建立神经网络模型

模型如图

![Image Name](https://cdn.kesci.com/upload/image/q17ipqoyrg.png?imageView2/0/w/960/h/960)

数学原理

![image-20210516211005881](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210516211005881.png)

建立神经网络的一般方法是

1.定义神经网络结构（输入单元数，隐藏单元数等）。
2.初始化模型的参数
3.循环：

- 实现前向传播
- 计算损失
- 后向传播以获得梯度
- 更新参数（梯度下降）

### 3.1  定义神经网络结构

定义三个变量：
   \- n_x：输入层的大小
   \- n_h：隐藏层的大小（将其设置为4）
   \- n_y：输出层的大小

```python
def layer_sizes(X, Y):
    n_x = X.shape[0] 
    n_h = 4
    n_y = Y.shape[0] 
    return (n_x, n_h, n_y)
```

### 3.2 初始化模型的参数

- 确保参数大小正确。
- 使用随机值初始化权重矩阵。
     \- 使用：`np.random.randn（a，b）* 0.01`随机初始化维度为（a，b）的矩阵。
- 将偏差向量初始化为零。
     \- 使用：`np.zeros((a,b))` 初始化维度为（a，b）零的矩阵。

```python
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
```

### 3.3 循环

#### 3.3.1 实现正向传播

计算Z[1],A[1],Z[2] 和 A[2] 

```python
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache
```

#### 3.3.2 计算cost

通过数学公式

![image-20210516211844129](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210516211844129.png)

计算

```python
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] 
    logprobs = Y*np.log(A2) + (1-Y)* np.log(1-A2)
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    return cost
```

#### 3.3.3 实现反向传播

公式如下图

![Image Name](https://cdn.kesci.com/upload/image/q17hcd4yra.png?imageView2/0/w/960/h/960)

```python
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2= A2 - Y
    dW2 = 1 / m * np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = 1 / m * np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads
```

3.3.4 实现梯度下降

关键是选择合适的学习速率，速度太慢可能会导致需要循环多次才能找到合适的参数值，而速率太快又会导致在一个区间往复。如图所示

较好的学习速率

![Image Name](https://cdn.kesci.com/upload/image/q17hh4otzu.gif?imageView2/0/w/960/h/960)

较差的学习速率

![Image Name](https://cdn.kesci.com/upload/image/q17hharbth.gif?imageView2/0/w/960/h/960)

```python
def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
```

3.3.4 在model集成函数

神经网络模型必须以正确的顺序组合先前构建的函数。

```python
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)"grads".
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters
```

## 四、预测

### 4.1 预测训练集

使用模型通过构建predict()函数进行预测，使用正向传播来预测结果。

```python
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions
```

运行代码预测实现数据

```python
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
```

结果如下图

![img](https://cdn.kesci.com/rt_upload/CB068684F93C4A2A8AE816EB492CDCBE/q17hj9pr80.png)

### 4.2 调整隐藏层数量

观察到不同大小隐藏层的模型的不同表现

```python
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
```

![img](https://cdn.kesci.com/rt_upload/F70D5A23097642688F6245327ACE9DD7/q17hkk98sl.png)

通过观察发现

- 较大的模型（具有更多隐藏的单元）能够更好地拟合训练集，直到最终最大的模型过拟合数据为止。
- 隐藏层的最佳大小似乎在n_h = 5左右。此值很好地拟合了数据，而又不会引起明显的过度拟合。
- 可以通过正则化，构建更大的模型（例如n_h = 50）而不会过度拟合。

## 五、总结

- 建立具有隐藏层的完整神经网络
- 善用非线性单位
- 实现正向传播和反向传播，并训练神经网络
- 了解不同隐藏层大小（包括过度拟合）的影响。