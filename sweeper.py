#!/usr/bin/env python
import collections
import unittest


# ----------------------------------------------------------------------------
# PARAMETER SWEEPER
# ----------------------------------------------------------------------------
class ParameterSweeper(object):
  """
  Randomly sample free parameters for fine tuning optimization algorithms

  Description
  ===========
  ParameterSweeper is an iterable object designed to simultanesouly
  sample multiple parameters over their range of feasible values. This
  is particularly useful in algorithm tuning where you want to find the
  optimal values of a set of free parameters.

  Random sampling is preferable to grid search, since parameters are not
  are redundantly sampled. For fast-changing parameters especially, this
  gives far better resolution of local minima.

  Args
  ====
    samples  - The number of samples to draw. Defaults to infinte.
               NOTE: the default value when using the object in an iterator
               context will cause an infinite loop. This can be useful when
               sampling until some criterion is met, but it is worth
               being aware of.

    optimize - If a finite number of samples is chosen and optimize is True,
               ParameterSweeper will solve a fast travelling salesman problem
               over the sample space so that consecutive samples are as
               close together as possible. This is particularly useful for
               algorithms that can be "warm started", where the previous
               optima is used as a starting point for the new optimzation.
               In such a case, having parameters close together in consecutive
               optimzations can lead to faster convergence.
               NOTE: samples must be < 10,000

    **kwargs - key=value pairs of the parameter names and a distribution
               generator. For example, if a parameter 'C' is uniformly
               distributed in the range [0, 1), then we could call,
                 ps = ParameterSweeper(C=lambda: numpy.random.uniform(0, 1))

  Returns
  =======
  An iterable that produces named tuples with fields in CASE-INSENSITIVE
  LEXICOGRAPHICAL ORDER. This is because the order of kwargs passed to the
  constructor is never known. However, with this strict ordering known,
  tuple unpacking can be reliably used in code (see the examples following)

  Examples
  ========

    # 1. Sample 400 values and iterate using named tuples
    ps = sweeper.ParameterSweeper(400,
            C = lambda: random.choice([0, 1, 2, 3, 4]),
            x = lambda: numpy.random.uniform(0, 1),
            y = lambda: numpy.random.standard_normal())

    for sample in ps:
      minima = optimize(data, sample.x, sample.y, sample.C)

    ---------

    2. Sample infinite values and iterate using tuple unpacking
    ps = sweeper.ParameterSweeper(
            C = lambda: random.choice([0, 1, 2, 3, 4]),
            x = lambda: numpy.random.uniform(0, 1),
            y = lambda: numpy.random.standard_normal())

    for C, x, y in ps:
      minima = optimize(data, x, y, C)
      if minima < eps:
        break
  """

  def __init__(self, samples=float('inf'), optimize=False, **kwargs):
    self.n = 0
    self.M = len(kwargs)
    self.N = samples
    self.optimize = optimize

    # create a named tuple of the parameters
    self.keys = sorted(kwargs.keys(), key=lambda s: s.lower())
    self.Parameters = collections.namedtuple('Parameters', self.keys)

    # compute the sample generators
    if optimize:
      # travelling salesman
      if samples > 10000:
        raise ValueError('The number of samples must be < 10,000 to solve the TSP')
      tsp = TravellingSalesman()
      self.generators, _ = tsp.optimize(samples, kwargs)
      self.get_sample    = self._optimized_sample
    else:
      # compute on the fly
      self.generators = kwargs
      self.get_sample = self._sample

  def __iter__(self):
    return self

  def next(self):
    if self.finished:
      raise StopIteration
    else:
      sample = self.get_sample()
      self.n += 1
      return sample

  def reset(self):
    self.n = 0

  @property
  def finished(self):
    return self.n >= self.N

  def _sample(self):
    return self.Parameters._make(self.generators[key]() for key in self.keys)

  def _optimized_sample(self):
    return self.Parameters._make(self.generators[self.n, m] for m in xrange(self.M))


# ----------------------------------------------------------------------------
# TRAVELLING SALESMAN
# ----------------------------------------------------------------------------
def has_numpy():
  try:
    import numpy as np
    return True, np
  except:
    return False, None

class TravellingSalesman(object):
  def __init__(self):
    available, self.np = has_numpy()
    if not available:
      raise ImportError('Numpy is required to solve the TSP')

  def optimize(self, samples, parameters):

    np = self.np
    M = samples                     # rows
    N = len(parameters)             # cols
    X = np.empty((M, N), order='C') # data matrix
    y = [0]                         # permutation matrix
    r = range(1,M)                  # residual matrix
    i = 0                           # current city

    # fill data matrix
    for n, key in enumerate(parameters):
      for m in xrange(M):
        X[m,n] = parameters[key]()

    # compute permutation
    while r:
      i = np.argmin(np.linalg.norm(X[i,:] - X[r,:], axis=1))
      i = r.pop(i)
      y.append(i)

    # permute the sampled data matrix
    return X[y,:], y


# ----------------------------------------------------------------------------
# UNIT TESTS
# ----------------------------------------------------------------------------
class TestParameterSweeper(unittest.TestCase):
  import random

  def test_sequence_order(self):
    g = (x for x in [1, 2])
    ps = ParameterSweeper(2, x=g.next)
    self.assertTrue(ps.next().x == 1)
    self.assertTrue(ps.next().x == 2)

  def test_sequence_terminates(self):
    ps = ParameterSweeper(1, x=self.random.random)
    ps.next()
    self.assertRaises(StopIteration, ps.next)

  def test_zero_samples(self):
    ps = ParameterSweeper(0, x=self.random.random)
    self.assertRaises(StopIteration, ps.next)

  def test_negative_samples(self):
    ps = ParameterSweeper(-10, x=self.random.random)
    self.assertRaises(StopIteration, ps.next)

  def test_reset(self):
    ps = ParameterSweeper(1, x=self.random.random)
    self.assertFalse(ps.finished)
    ps.next()
    self.assertTrue(ps.finished)
    ps.reset()
    self.assertFalse(ps.finished)

  def test_named_tuple_unpacking(self):
    xg = (x for x in [1])
    yg = (x for x in [2])
    Cg = (x for x in [3])
    ps = ParameterSweeper(1, y=yg.next, x=xg.next, C=Cg.next)
    C, x, y = ps.next()
    self.assertTrue(x == 1 and y == 2 and C == 3)

@unittest.skipIf(not has_numpy(), 'Travelling Salesman requires numpy')
class TestTravellingSalesman(unittest.TestCase):
  import random

  def test_permutation_is_unique(self):
    tsp  = TravellingSalesman()
    X, y = tsp.optimize(100, {'x': self.random.random, 'y': self.random.random})
    self.assertTrue(len(y) == len(set(y)))


# ----------------------------------------------------------------------------
# COMMAND LINE
# ----------------------------------------------------------------------------
def _inline_sample(parameters):
  """
  Generate random sample from a named numpy.random distribution
  """
  import numpy.random
  str_rep = ''
  for parameter, args in parameters.items():
    distribution = args[0]
    args = [float(arg) if '.' in arg else int(arg) for arg in args[1:]]
    val  = getattr(numpy.random, distribution)(*tuple(args))
    str_rep = ' '.join([str_rep, parameter, str(val)])
  return str_rep


if __name__ == '__main__':
  import sys
  import argparse

  # argument parser
  parser = argparse.ArgumentParser(
              description='Randomly sample free parameters for fine tuning optimization algorithms',
              add_help=False)

  parser.add_argument('-h', '--help', action='store_true', help='show detailed docstring')
  parser.add_argument('-t', '--test', action='store_true', help='run the tests')
  parser.add_argument('-i', '--inline', action='store_true', help='sample parameters on the command line')
  args, unknown = parser.parse_known_args()

  # print docstring
  if args.help:
    help(ParameterSweeper)

  # run tests
  elif args.test:
    unittest.main(argv=sys.argv[1:])

  # output sample of parameters to command line
  elif args.inline:
    parameters = {}
    key = None
    for arg in unknown:
      if arg.startswith('-'):
        key = arg
        parameters[key] = []
      else:
        parameters[key].append(arg)
    print(_inline_sample(parameters))

  # at least one argument is required
  else:
    parser.print_help()
