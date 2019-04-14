---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.5
  kernelspec:
    display_name: Python [default]
    language: python
    name: python3
---

# Thinking in tensors, writing in PyTorch

A hands-on course by [Piotr Migdał](https://p.migdal.pl) (2019).

## Notebook 2: Gradient descent


Open in Colab: https://colab.research.google.com/github/stared/thinking-in-tensors-writing-in-pytorch/blob/master/2%20Gradient%20Descent.ipynb

> X: I want to learn Deep Learning.  
> Me: Do you know what is gradient?  
> X: Yes  
> Me: Then, it an easy way downhill!

Memic content:

* https://twitter.com/jebbery/status/995491957559439360
* https://twitter.com/smaine/status/994723834434502658


**CONTENT MORE OR LESS THERE, NEEDS DESCRIPTIONS**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

$$y = x^2$$

$$ \frac{\partial y}{\partial x} = 2 x$$

For $x^2$ we can calculate it:

$$\lim_{x \to 0} \frac{y(x + h) - y(x)} {h}$$

Limit is a mathematical tool for 

$$\frac{x^2 + 2 x h + h^2 - x^2}{h} = 2 x + h $$

```python
X = np.linspace(-4, 4, num=100)
Y = X**2
```

```python
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,
                               sharex=True, figsize=(7, 4))

ax0.plot(X, Y)
ax0.set(title='', xlabel='', ylabel='y')

ax1.plot(X, 2 * X)
ax1.set(title='', xlabel='x', ylabel='dy/dx')

fig.tight_layout()
```

## Numerical derivative in NumPy

```python
# we can go it automatically
plt.plot((X[1:] + X[:-1]) / 2, np.diff(Y) / np.diff(X))
```

## Symbolic derivative in PyTorch

```python
x = torch.tensor(10., requires_grad=True)
y = x.pow(2)
y.backward()

# y
y

# dy / dx
x.grad
```

## Gradient descent

```python
lr = 0.2
x0 = 4.

xs = [x0]
x = torch.tensor(x0, requires_grad=True)

for i in range(10):
    y = x.pow(2)
    y.backward()
    x.data.add_(- lr * x.grad.data)
    x.grad.data.zero_()
    xs.append(x.item())

xs
```

```python
points_X = np.array(xs)
points_Y = points_X**2

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
ax.plot(X, Y)
ax.plot(points_X, points_Y, '-')
ax.plot(points_X, points_Y, 'r.')
ax.set(title='Gradient descent', xlabel='x', ylabel='y');
```

## Exercise

Try other learning rates, e.g.:

* 0.1
* 0.5
* 0.75
* 1.
* 1.5
* -0.5


## Slightly more complicated functions

```python
def f(x):
    return x - 4 * x**2 + 0.25 * x**4

def df(x):
    return 1 - 8 * x + x**3
```

```python
X = np.linspace(-4, 4, num=100)

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,
                               sharex=True, figsize=(7, 4))

ax0.plot(X, f(X))
ax0.set(title='', xlabel='', ylabel='y')

ax1.plot(X, df(X))
ax1.hlines(y=0, xmin=-4, xmax=4, linestyles='dashed')
ax1.set(title='', xlabel='x', ylabel='dy/dx')

fig.tight_layout()
```

```python
lr = 0.1
x0 = 4.

xs = [x0]
x = torch.tensor(x0, requires_grad=True)

for i in range(10):
    y = f(x)
    y.backward()
    x.data.add_(- lr * x.grad.data)
    x.grad.data.zero_()
    xs.append(x.item())

xs
```

```python
points_X = np.array(xs)
points_Y = f(points_X)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
ax.plot(X, f(X))
ax.plot(points_X, points_Y, '-')
ax.plot(points_X, points_Y, 'r.')
ax.set(title='Gradient descent', xlabel='x', ylabel='y');
```

## Gradient in 2d

Gradients make sense for more dimensions. For mountains, gradient would be a vector directed along to the biggest slope.



For example, let's have a function:

$$y = g(x_1, x_2) = x_1^2 + \sin(x_2)$$

In this case, gradient is a vector. To calculate gradient we use [partial derviative](https://en.wikipedia.org/wiki/Partial_derivative).

$$\nabla g = \left( \frac{\partial g}{\partial x_1}, \frac{\partial g}{\partial x_2} \right) = \left( 2 x_1, \cos(x_2) \right) $$


Gradient symbol:

$$\nabla = \left( \frac{\partial }{\partial x_1}, \frac{\partial }{\partial x_2} \right)$$

```python
X0_ = np.linspace(-4, 4, num=100)
X1_ = np.linspace(-4, 4, num=100)
X0, X1 = np.meshgrid(X0_, X1_)
```

```python
# purely technically, so that we have the same code
# for NumPy and PyTorch

def sin(x):
    if type(x) == torch.Tensor:
        return x.sin()
    else:
        return np.sin(x)
    
def cos(x):
    if type(x) == torch.Tensor:
        return x.cos()
    else:
        return np.cos(x)

# now the functions and their gradients
# (calculated by hand)
    
def g(x0, x1):
    return 0.25 * x0**2 + sin(x1)

def dg_dx0(x0, x1):
    return 0.5 * x0

def dg_dx1(x0, x1):
    return cos(x1)
```

Or, let's draw a [contour plot](https://en.wikipedia.org/wiki/Contour_line), well known from topographic maps.

```python
cs = plt.contour(X0, X1, g(X0, X1), cmap='coolwarm')
plt.clabel(cs, inline=1, fontsize=10)
plt.title("g")
plt.xlabel("x0")
plt.ylabel("x1");
```

```python
cs = plt.contour(X0, X1, dg_dx0(X0, X1), cmap='coolwarm')
plt.clabel(cs, inline=1, fontsize=10)
plt.title("dg/dx0")
plt.xlabel("x0")
plt.ylabel("x1");
```

```python
cs = plt.contour(X0, X1, dg_dx1(X0, X1), cmap='coolwarm')
plt.clabel(cs, inline=1, fontsize=10)
plt.title("dg/dx1")
plt.xlabel("x0")
plt.ylabel("x1");
```

```python
X0_less = X0[::5, ::5]
X1_less = X1[::5, ::5]

fig1, ax1 = plt.subplots()
ax1.set_title(r'$\nabla g$')
Q = ax1.quiver(X0_less, X1_less,
               dg_dx0(X0_less, X1_less), dg_dx1(X0_less, X1_less),
               units='width')
```


## Gradient descent in 2d

```python
lr = 0.25
v = [-3.5, 1.0]

xs = [v[0]]
ys = [v[1]]
v = torch.tensor(v, requires_grad=True)

for i in range(20):
    y = g(v[0], v[1])
    y.backward()
    v.data.add_(- lr * v.grad.data)
    v.grad.data.zero_()
    
    xs.append(v[0].item())
    ys.append(v[1].item())
```

```python
cs = plt.contour(X0, X1, g(X0, X1), cmap='coolwarm')
plt.clabel(cs, inline=1, fontsize=10)
plt.plot(xs, ys, '-')
plt.plot(xs, ys, 'o')
```

```python

```
