Parameter Sweeper
=================
Randomly sample free parameters for fine tuning optimization algorithms

Introduction
------------
When tuning free algorithm parameters, random sampling over the space is preferrable to grid search, since parameters are not redunantly sampled. For fast changing parameters especially, this gives far better resolution of local minima.

This small Python class provides a clean interface for sampling multiple parameters.

Example
-------

```python
# 1. Sample 400 values and iterate using named tuples
ps = sweeper.ParameterSweeper(400,
        C = lambda: random.choice([0, 1, 2, 3, 4]),
        x = lambda: numpy.random.uniform(0, 1),
        y = lambda: numpy.random.standard_normal())

for sample in ps:
  minima = optimize(data, sample.x, sample.y, sample.C)

---------

# 2. Sample infinite values and iterate using tuple unpacking
ps = sweeper.ParameterSweeper(
        C = lambda: random.choice([0, 1, 2, 3, 4]),
        x = lambda: numpy.random.uniform(0, 1),
        y = lambda: numpy.random.standard_normal())

for C, x, y in ps:
  minima = optimize(data, x, y, C)
  if minima < eps:
    break

---------

# 3. Sample on the command line
python sweeper.py --inline -x uniform 0 1 -y standard_normal | xargs python optimize.py
```

Travelling Salesman Optimization
--------------------------------
If a finite number of samples is chosen, ParameterSweeper offers the option to solve a fast travelling salesman problem over the sample space so that consecutive samples are as close together as possible. This is particularly useful for algorithms that can be "warm started", where the previous optima is used as a starting point for the new optimzation. In such a case, having parameters close together in consecutive optimzations can lead to faster convergence.
