# Copyright (c) 2022, emece67 - MIT License
#!/usr/bin/env python3

import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import solve
import matplotlib.pyplot as plt
import warnings

def hornersparse(x, powers, coeffs):
  '''hornersparse evaluates a polynomial using Horner's scheme managing
      the case of some, or many, coefficients being 0.
      hornersparse(x, powers, coeffs) evaluates at points `x` a polynomial
      with (integer) powers given in `powers` and coefficients given in
      `coeffs`. Both `powers` and `coeffs` must have the same number
      of elements and `powers` must be a strictly increasing sequence.
      All of `x`, `powers` and `coeffs` can be lists or numpy.arrays,
      whereas the return value is always a numpy.array.

    Usage example:
      * y = hornersparse(x, [0, 1, 3, 15], [7, 9, 4, 5])

        will evaluate polynomial y = 7 + 9x + 4x^3 + 5x^15 at points x.'''

  # convert to np.array in case of lists...
  x = np.array(x)
  powers = np.array(powers)
  coeffs = np.array(coeffs)

  if powers.shape != coeffs.shape:
    raise ValueError('Both powers and coefficients lists/arrays must be of same shape')

  powersi = np.rint(powers).astype(int)
  if not all(np.isclose(powers, powersi)):
    raise ValueError('Powers must be integers')

  diff_powers = np.flip(np.diff(powersi))
  if any(diff_powers <= 0):
    raise ValueError('The sequence of powers must be strictly increasing')

  result = np.full_like(x, coeffs[-1])
  coeffs = np.flip(coeffs[:-1])
  for diff_power, coeff in zip(diff_powers, coeffs):
    result = coeff + np.power(x, diff_power) * result

  if powersi[0]:
    result *= np.power(x, powersi[0])

  return result


def remezfit(f, a, b, degree_powers, relative = False, odd = False, even = False, arg = None, weight = None, dtype = np.double, trace = False):
  '''remezfit fits a polynomial to a function in an equi-ripple sense.
      remezfit(f, a, b, degree_powers, [, option...]) finds the
      coefficients of a polynomial P of degree `degree_powers` that fits
      the function `f` over interval (`a`, `b`) in an equi-ripple sense.
      The default behaviour is to fit the polynomial with an equi-ripple
      _absolute_ error.
    The parameter `degree_powers` can also be a list or array containing
      the desired powers in the polynomial.
    This function accepts some options (that can be entered as named
      parameters), they are:
        * `relative`: if True, the approximation shows an equi-ripple
                      _relative_ error behaviour, otherwise it works
                      with the absolute error;
        * `odd`:      set to True if it is known that the function to
                      be fitted shows odd symmetry around the left
                      extreme of the interval;
        * `even`:     set to True if it is known that the function to
                      be fitted shows even symmetry around the left
                      extreme of the interval;
        * `arg`:      fits a polynomial not in x, but in arg(x),
                      being arg a function, e.g.:
                        arg = lambda x: (x - 1) / (x + 1)
                      will fit a polynomial in (x - 1) / (x + 1);
        * `weight`:   specifies the function used to weight the error;
        * `dtype`:    sets the data type used in calculations;
        * `trace`:    shows the error plot resulting after each
                      iteration.

        Note that options `relative` and `weigth` cannot be specified
          simultaneously, as both `odd` and `even` options.

        If any of options `odd` or `even` is specified and
          `degree_powers` is an integer (specifying the degree of the
          polynomial to fit), then the fitted polynomial will be also
          odd or even, respectively.

    p, prec = remezfit(f, a, b, degree_powers, [, option...]) returns in
      `p` the fitted polynomial coefficients ---starting from the lowest
      order coefficient, thus suitable to be used by
      remezfit.hornersparse() above---. The `prec` returned value is:
        * the maximum weighted absolute error, when option relative is
            False
        * the minimum number of correct digits, otherwise

    Usage examples:
      * p, prec = remezfit(lambda x: np.sin(x), 0, np.pi/2, 5, relative = True, odd = True, dtype = np.single, trace = True)

        will fit an odd, 5th degree, polynomial in x to sin(x) along
          interval (0, pi/2) minimizing maximum relative error. Will
          use single precision numbers during calculations and will
          show the error plot after each iteration. `p` will contain
          the polynomial coefficients and `prec` the number of correct
          digits of the approximation.

      * p, prec = remezfit(lambda x: (np.cos(x) - 1)/np.power(x, 2), 1e-15, np.pi/2, [0, 2, 4], even = True, weight = lambda x: np.power(x, 2))

        will fit a polynomial in x with with powers {0, 2, 4} to
          (cos(x) - 1)/x^2 along interval (1e-15, pi/2) minimizing
          maximum weighted (by x^2) error. `p` will contain the
          coefficients for powers {0, 2, 4} and `prec` the maximum
          weighted error of the approximation.

      * p, prec = remezfit(lambda x: np.log(x), 1, np.sqrt(2), 3, odd = True, arg = lambda x: (x - 1) / (x + 1))

        will fit an odd, 3rd degree, polynomial in (x - 1) / (x + 1)
          to log(x) along interval (1, sqrt(2)) minimizing the maximum
          absolute error. `p` will contain the polynomial coefficients
          and `prec` the maximum absolute error of the approximation.'''

  # sanity checks
  if not np.isscalar(a) or not np.isscalar(b):
    raise TypeError('Both a and b parameters must be scalars')

  if a == b:
    raise ValueError('Zero length interval')

  if even and odd:
    raise ValueError('Cannot specify both odd and even options simultaneously')

  if relative and (weight != None):
    raise ValueError('Cannot specify both relative and weight options simultaneously')

  ni = np.rint(degree_powers).astype(int)
  if np.isscalar(ni):
    if not np.isclose(ni, degree_powers):
      raise ValueError('Degree of polynomial must be an integer')

    if ni < 0:
      raise ValueError('Cannot approximate with a negative degree polynomial')

    if even and ni <= 1:
      raise ValueError('Even order polynomial fit requires degree > 1')

    powers = np.array([*range(1 if odd else 0, ni + 1, 2 if even or odd else 1)])
  else:
    if not all(np.isclose(ni, degree_powers)):
      raise ValueError('Powers of polynomial must be integers')

    if any(ni < 0):
      raise ValueError('Powers of polynomial must be >= 0')

    powers = np.unique(ni)

    if (1 == powers.size) and (0 == powers[0]):
      raise ValueError('Cannot fit into a constant polynomial')

  if a > b:
    a, b = b, a
  a, b = np.array((a, b), dtype = dtype)

  # number of coefficients in the target polynomial
  n_coeffs = len(powers)

  # number of extrema, also number of equations in the linear system
  k = n_coeffs + 1

  # number of extrema for the whole interval (reflected for odd/even)
  if odd:
    k_w = 2 * k
  elif even:
    k_w = 2 * k - 1
  else:
    k_w = k

  # initial positions of the extrema (peaks of n+1 Chebyshev polynomial)
  c = np.cos(np.linspace(np.pi, 0, k_w, endpoint = True, dtype = dtype))
  if odd or even:
    c = c[k_w - k :]
    x = a + (b - a) * c
  else:
    x = (a + b) / 2 + (b - a) * c / 2

  # number of samples
  m = min(512, 32*np.ceil((b - a) / (x[-1] - x[-2])).astype(int))

  # minimum required distance between maxima when looking for peaks
  min_peak_dist = 24

  # indices for the error peaks
  idx = np.full(k, -m)

  # all values of x, arg(x), f(x) and weight(x)
  x_plot = np.linspace(a, b, m, endpoint = True, dtype = dtype)
  vx_plot = arg(x_plot) if arg else x_plot
  f_plot = f(x_plot)
  if weight != None:
    w_plot = weight(x_plot)
  else:
    w_plot = np.ones_like(x_plot)

  # the (now empty) coefficient matrix for the linear system
  m_a = np.zeros((k, k), dtype = dtype)

  # alternating +1, -1 column (coeffs of the error term in the linear system)
  pm1 = np.ones(k, dtype = dtype)
  pm1[1::2] = -1

  # best error of the iterations
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    e_best = np.finfo(dtype).max

  # number of iterations
  n_iter = 4 * k

  # results for each iteration
  idxs = np.zeros((n_iter, k), dtype = int)

  # iterate
  iter = 0
  while iter < n_iter:
    # build the constant matrix
    m_b = f(x)
    # build the coefficient matrix
    vx = arg(x) if arg else x
    m_a[:, 0] = np.power(vx, powers[0])
    for i in range(1, k - 1):
      m_a[:, i] = m_a[:, i - 1] * np.power(vx, powers[i] - powers[i - 1])
    m_a[:, -1] = pm1 * m_b if relative else pm1
    if weight:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_a[:, -1] /= weight(x)
        m_a[:, np.isnan(m_a[:, -1])] = 16*max(m_a[:, -1])
    # compute new polynomial
    p = solve(m_a, m_b)[:-1].astype(dtype)
    # compute error
    e_plot = (hornersparse(vx_plot, powers, p) - f_plot) * w_plot
    if relative:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_plot = e_plot / f_plot
    e_plot[np.isnan(e_plot)] = 0
    e_max = max(abs(e_plot))
    # get new locations of extrema
    idx_new = find_peaks(abs(e_plot), distance = min_peak_dist)[0].astype(int)
    peaks_found = idx_new.size
    # add interval extremes, if needed
    if peaks_found == k - 2 and idx_new[0] != 0 and idx_new[-1] != m - 1:
      idx_new = np.hstack((0, idx_new, m - 1))
    elif peaks_found == k - 1:
      if abs(e_plot[0]) < abs(e_plot[-1]) and idx_new[-1] != m - 1:
        idx_new = np.hstack((idx_new, m - 1))
      elif idx_new[0] != 0:
        idx_new = np.hstack((0, idx_new))
    elif peaks_found > k:
      idx_new = idx_new[-k:]
    # sanity check
    if odd and (0 == idx_new[0]):
      idx_new[0] = np.random.randint(0.2 * idx_new[1], 0.8 * idx_new[1])
    # sanity check
    if k != idx_new.size:
      if k > idx_new.size:
        print('Iteration exited because could not locate enough error extrema')
      if 0 == iter:
        p_best = p
      break
    # convergence tests
    repeated = np.any(np.all(idxs == idx_new, axis = 1))
    idxs[iter, :] = idx_new
    if repeated or (0 == max(abs(idx - idx_new))) or ((1 == max(abs(idx - idx_new))) and (e_max <= e_best)):
      iter = n_iter
    idx = idx_new
    if e_max < e_best:
      e_best = e_max
      p_best = p
    # plot intermediate errors
    if trace:
      print('e_max = %e' % e_max)
      plt.clf()
      if relative:
        plt.plot(x_plot, e_max * f_plot, "r")
        plt.plot(x_plot, -e_max * f_plot, "r")
      else:
        plt.plot(x_plot, e_max * np.ones(m), "r")
        plt.plot(x_plot, -e_max * np.ones(m), "r")
      plt.plot(x_plot, (hornersparse(vx_plot, powers, p) - f_plot) * w_plot, "b")
      plt.plot(x_plot[idx], (hornersparse(vx_plot[idx], powers, p) - f_plot[idx]) * w_plot[idx], "xg")
      plt.xlim((a, b))
      plt.grid(True)
      plt.pause(1)
    # prepare next iteration
    x = x_plot[idx]
    iter += 1

  # iteration ended, compute `prec` value
  e_plot = (hornersparse(vx_plot, powers, p_best) - f_plot) * w_plot
  if relative:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      e_plot = e_plot / f_plot
  e_plot[np.isnan(e_plot)] = 0
  if relative:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      digits = abs(np.log10(abs(e_plot)))
    prec = min(digits)
  else:
    prec = max(abs(e_plot))

  # plot final results
  if trace:
    plt.clf()
    if relative:
      plt.plot(x_plot, prec * np.ones(m), "r")
      plt.plot(x_plot, digits, "b")
    else:
      plt.plot(x_plot, prec * np.ones(m), "r")
      plt.plot(x_plot, -prec * np.ones(m), "r")
      plt.plot(x_plot, e_plot, "b")
    plt.xlim((a, b))
    plt.grid(True)
    plt.pause(1)

  return p, prec


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
    description = 'Fits a polynomial to a function in an equi-ripple sense.',
    epilog = 'Returns the polynomial coefficients (in increasing order) and the minimum number of correct digits if --relative is given, the maximum weighted absolute error otherwise. When specifying the arguments {f, a, b, degree_powers, ARG, WEIGHT}, it can be assumed that package numpy is imported as np.')
  parser.add_argument('f',
    help = 'function to be fitted (may be a " quoted string with a lambda expression)')
  parser.add_argument('a',
    help = 'left extreme of the interval where the function will be fitted')
  parser.add_argument('b',
    help = 'right extreme of the interval where the function will be fitted')
  parser.add_argument('degree_powers',
    help = 'degree of the polynomial (when scalar) or powers in the polynomial (when array)')
  parser.add_argument('-r', '--relative', action = 'store_true',
    help = 'fits minimizing the maximum relative error, otherwise minimizes the maximum absolute error')
  parser.add_argument('-o', '--odd', action = 'store_true',
    help = 'the function to fit does show odd symmetry (around the left interval extreme)')
  parser.add_argument('-e', '--even', action = 'store_true',
    help = 'the function to fit does show even symmetry (around the left interval extreme)')
  parser.add_argument('-a', '--arg', default = 'None',
    help = 'if specified, it must be a function, then the argument for the polynomial will be arg(x) instead of x (ARG may be a " quoted string with a lambda expression).')
  parser.add_argument('-w', '--weight', default = 'None',
    help = 'if specified, it must be a function, then the error will be weighted by weight(x) (WEIGHT may be a " quoted string with a lambda expression).')
  parser.add_argument('-d', '--dtype', default = 'double', choices = ['half', 'single', 'double', 'longdouble'],
    help = 'the float type to be used in computations')
  parser.add_argument('-t', '--trace', action = 'store_true',
    help = 'show error plots after each iteration')
  args = parser.parse_args()

  p, prec = remezfit(
    f = eval(args.f),
    a = eval(args.a),
    b = eval(args.b),
    degree_powers = eval(args.degree_powers),
    relative = args.relative,
    odd = args.odd,
    even = args.even,
    arg = eval(args.arg),
    weight = eval(args.weight),
    dtype = args.dtype,
    trace = args.trace)

  np.set_printoptions(floatmode = 'unique')
  print('p = %s\n' % repr(p))
  print('prec = %e\n' % prec)
