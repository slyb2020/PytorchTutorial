# 用于测试pytorch的自动求导功能
import torch
from torch import nn

# 先讨论单步运算时的自动求导问题；
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)
# Build a computational graph.
y = w * x + b  # y = 2 * x + 3
# Compute gradients.
y.backward()
# Print out the gradients.
print(x.grad)  # x.grad = 2
print(w.grad)  # w.grad = 1
print(b.grad)  # b.grad = 1
# 上面这段程序说明了pytorch自动求导的基本概念：调用y的backward()方法，那么x.grad变量的物理意义是y对x的偏导
# 但是，y对x的偏导不会自动归零，pytorch把它设为自动累加状态。所以如果还用原来的w，x，b再计算一次y，
# 然后再调用一次y.backward(),就会发现偏导值被自动累加了，下面的程序说明了这个问题
y = w * x + b  # y = 2 * x + 3
y.backward()
print(x.grad)  # x.grad = 4
print(w.grad)  # w.grad = 2
print(b.grad)  # b.grad = 2
# 如何避免自动累加的问题呢？可以在第二次计算前先将x, w, b的grad清零, 代码如下，注意：grad因该是float类型
x.grad = torch.tensor(0.)
w.grad = torch.tensor(0.)
b.grad = torch.tensor(0.)
y = w * x + b  # y = 2 * x + 3
y.backward()
print(x.grad)  # x.grad = 2
print(w.grad)  # w.grad = 1
print(b.grad)  # b.grad = 1

# 上面的代码x,w,b都是标量（scalar），这是对pytorch来讲是特殊情况，pytorch一般处理的都是矩阵数据，所以：
# 我们再看下面一段代码：
a = torch.randn(size=(1, 1, 3), requires_grad=True)
b = torch.randn(size=(1, 1, 3), requires_grad=True)
print("a=", a)
print("b=", b)
c = (a + b)
print("c=", c)
# 先给a,b,c赋好值，然后我们再求c对a或b的偏导，如下的代码可以吗？
# c.backward()
# print("a.grad=", a.grad)
# print("b.grad=", b.grad)
# 看来是不行的，为什么呢？因为pytorch只能对标量求偏导。这是pytorch的一个很重要的概念
# 所以我们先将上面的代码注释掉，然后做如下修改：
c = a + b
c[0, 0, 0].backward()
print("a.grad=", a.grad)
print("b.grad=", b.grad)
# c[0,0,0]是标量，这就可以求导了，但是现在a.grad和b.grad的物理意义是什么？
# 当然还是c[0,0,0]对a和b的偏导数了。
# 到这里我们还应该弄明白一个问题：a.grad和b.grad应该是什么形状？
# 很显然，a.grad的形状应该和a的形状完全相同。因为a.grad是a的每一个元素对c的偏导数。


# 到此，应该对单步运算时的自动求导问题弄明白了，
# 下面讨论多步运算时的自动求导问题：链式求导法则
# 还是先看一下标量运算的情况：
a = torch.tensor(3., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(1., requires_grad=True)
d = a + b
print("d=", d)
e = d * c
print("e=", e)
e.backward()
print("a.grad=", a.grad)
print("b.grad=", b.grad)
print("c.grad=", c.grad)
# 根据结果，我们可以进一步理解链式求导法则，以及pytorch自动求导对链式法则的实现过程
# 但是这里还有几个问题需要理解：d作为中间变量是否由d.grad?e作为结果，是否有e.grad?我们不妨通过下面的代码试一下：
# print(d.grad)
# print(e.grad)
# 我们看到打印出来的都是None，同时收到警告信息。警告里提到，只有所谓的leaf张量才有能够对其.grad属性进行读写，如果不是leaf就不能读写.grad
# 那么什么是leaf张量？
print("a is leaf:", a.is_leaf)
print("b is leaf:", b.is_leaf)
print("c is leaf:", c.is_leaf)
print("d is leaf:", d.is_leaf)
print("e is leaf:", e.is_leaf)
# 现在应该明白了：那些直接被赋值（即开头的张量）是leaf张量(不管其requires_grad属性是True还是False)。而那些中间张量都不是叶张量，
# 不是leaf张量就不要对其.grad进行读写操作，因此不要对其print
# 那么还有一个问题需要搞懂：中间张量是否有requires_grad属性？有的话其requires_grad是True还是False？不妨通过下面的代码试一下：
print("中间张量是否有.requires_grad属性？", d.requires_grad)
# 可以看到，不但有，而且自动被设置为True了
# 但是你却不能对其.grad属性进行print，这似乎有些矛盾，但事实就是如此。不必感到奇怪，我们要做的就是去理解。
# 我们不妨这么想：中间变量是在计算中产生的，它的grad值只与计算表达式及叶张量有关，该什么样就什么样，
# 我们不能去改变它，自然也就没必要去对其进行存取操作了。所以pytorch就没有提供对非叶张量的.grad属性进行操作的方法。
# 当然非叶张量的其它属性都是可以进行读写操作的，只是其.grad属性被保护了而已。

a.requires_grad = False
print("a是否是叶张量？", a.is_leaf)

"""
几点总结：
1. 自动求导的对象必须时float型，整数不行
2. 只能对标量求导，数组，矩阵都不行，也就是不能对非标量调用.backward()方法
3. 单步求导与这步进行的运算有关
4. 多步求导是每步导数的乘积，即链式求导法则
5. a啥样，(a.grad)也长啥样。指的是shape
6. 叶张量是起始端张量(被赋值的张量），不管其requires_grad属性为True还是False都是叶张量，都可以print其.grad属性
7. 中间张量都不是叶张量，都不能访问其.grad属性
8. 中间张量的require_grad属性默认被设置为True
9. 单看a.grad无法确定其物理含义，只知道它是某个张量对a的偏导数。要想明白是哪个张量的偏导数就要看是哪个张量调用了.backward()方法
10. 要牢记.grad属性具有自动累加功能。如果不想出现不必要的错误，一定要记得在适当的时候对其进行清零。
"""

# 如果对上述内容都懂了，那么作为检验，也该能很容易地用pytorch编写出一个线性回归分析程序了，试一下吧：

x = torch.randn(size=(10, 1))
label = x * 3.5 + 0.6
w = torch.randn(size=(1,), requires_grad=True)
b = torch.randn(size=(1,), requires_grad=True)

maxEpoch = 1000
Loss = torch.nn.MSELoss()
optim = torch.optim.SGD((w, b), lr=1e-3, momentum=0.9)
for epoch in range(maxEpoch):
    for i in range(10):
        optim.zero_grad()
        y = w * x[i] + b
        loss = Loss(y, label[i])
        loss.backward()
        optim.step()
print((w * 4.2 + b).item(), 3.5 * 4.2 + 0.6)

# 如果对上述的内容都弄明白了，那么现在可以受用pytorch真正构造一个简单的神经网络模型了：

x = torch.randn(size=(10, 3))
label = torch.randn(size=(10, 2))
linear = nn.Linear(3, 2)
print(linear.weight)
print(linear.bias)
Loss = torch.nn.MSELoss()
optim = torch.optim.SGD(linear.parameters(), lr=0.5, momentum=0.9)
optim.zero_grad()
y = linear(x)
print("y=", y)
print("label=", label)
loss = Loss(y,label)
print("loss=", loss.item())
print("x=",x)
loss.backward()
print("dloss/dw=", linear.weight.grad)
print("dloss/dbias=", linear.bias.grad)
optim.step()
optim.zero_grad()
y = linear(x)
loss = Loss(y,label)
print("loss2=", loss.item())
loss.backward()
print("dloss/dw=", linear.weight.grad)
print("dloss/dbias=", linear.bias.grad)
optim.step()
y = linear(x)
loss = Loss(y, label)
print("loss3=", loss.item())
optim.zero_grad()
loss.backward()
print("dloss/dw=", linear.weight.grad)
print("dloss/dbias=", linear.bias.grad)
optim.step()