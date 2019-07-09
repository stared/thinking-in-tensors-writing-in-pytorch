# Thinking in tensors, writing in PyTorch

A hands-on deep learning introduction, from pieces.

For an interactive, installation-free version, use Colab: https://colab.research.google.com/github/stared/thinking-in-tensors-writing-in-pytorch/

By [Piotr Migdał](https://p.migdal.pl/) et al. (Weronika Ormaniec, possibly others)


> “Study hard what interests you the most in the most undisciplined, irreverent and original manner possible.”  ― Richard Feynman

> “Scientists start out doing work that's perfect, in the sense that they're just trying to reproduce work someone else has already done for them. Eventually, they get to the point where they can do original work. Whereas hackers, from the start, are doing original work; it's just very bad. So hackers start original, and get good, and scientists start good, and get original.” - Paul Graham in [Hackers and Painters](http://www.paulgraham.com/hp.html)

## What's that?

Mathematical concepts behind deep learning using PyTorch 1.0.

* All math equations as PyTorch code
* Explicit, minimalistic examples
* Jupyter Notebook for interactivity
* “On the shoulders of giants” - I link and refer to the best materials I know
* Fully open source & open for collaboration (I guess I will go with MIT for code, CC-BY for anything else)


## Why not something else?

There are quite a few practical introductions to deep learning. I recommend [Deep Learning in Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet (the Keras author). Or you want, you can classify small pictures, or extraterrestrial beings, today.

When it comes to the mathematical background, [Deep Learning Book](https://www.deeplearningbook.org/) by Ian Goodfellow et al. is a great starting point, giving a lot of overview. Though, it requires a lot of interest in maths. Convolutional networks start well after page 300.

I struggled to find something in the middle ground - showing mathematical foundations of deep learning, step by step, at the same time translating it into code. The closest example is [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) (which is, IMHO, a masterpiece). Though, I believe that instead of using NumPy we can use PyTorch, giving a smooth transition between mathematic ideas and a practical, working code.

Of course, there are quite a few awesome posts, notebooks and visualizations. I try to link to the ones that are useful for reader. In particular, I maintain a collaborative list of [Interactive Machine Learning, Deep Learning and Statistics websites](https://p.migdal.pl/interactive-machine-learning-list/).


## Contribute!

Crucially, this course is for you, the reader. If you are interested in one topic, let us know! There is nothing more inspiring that eager readers.


## Style

* Start with concrete examples first
* First 1d, then more
* Equations in LaTeX AND PyTorch
* `x.matmul(y).pow(2).sum()` not `torch.sum(torch.matmul(x, y) ** 2)`


## Adverts

A few links of mine:

* My deep learning framework credo: [Keras or PyTorch as your first deep learning framework](https://deepsense.ai/keras-or-pytorch/)
* [Keras vs. PyTorch: Alien vs. Predator recognition with transfer learning ](https://deepsense.ai/keras-vs-pytorch-avp-transfer-learning/)
* [My general overview of “how to start data science”](https://p.migdal.pl/2016/03/15/data-science-intro-for-math-phys-background.html) (turns out - not only for math/phys background; though, I intend to write a separate text for non-STEM backgrounds)
* I am an independent AI consultant, specializing in giving hands-on trainings in deep learning (and general machine learning). If you are interested in a workshop, let me know at [p.migdal.pl](https://p.migdal.pl/)!
