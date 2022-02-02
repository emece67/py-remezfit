# py-remezfit

A direct and basic implementation of the Remez algorithm to fit polynomials to functions in an equi-ripple sense (or minimax). Wrote in Python using `numpy`, `scipy` and `matplotlib`.

There is a single Python file here `remezfit.py` providing module `remezfit`. This module:

* Provides function `remezfit.remezfit()` that fits a polynomial to a function in an equi-ripple sense. `remezfit(f, a, b, degree_powers, [, option...])` finds the coefficients of a polynomial P of degree `degree_powers` that fits the function `f` over interval (`a`, `b`) in an equi-ripple sense. The default behaviour is to fit the polynomial with an equi-ripple _absolute_ error. The parameter `degree_powers` can also be a list or array containing the desired powers in the polynomial. This function accepts some options (that can be entered as named parameters), they are:

  * `relative`: if `True`, the approximation shows an equi-ripple _relative_ error behaviour, otherwise it works with the absolute error;
  * `odd`: set it to `True` if it is known that the function to be fitted shows odd symmetry around the left extreme of the interval;
  * `even`: set it to `True` if it is known that the function to be fitted shows even symmetry around the left extreme of the interval;
  * `arg`: fits a polynomial not in $x$, but in $arg(x)$, being `arg` a function, e.g.:
    `arg = lambda x: (x - 1) / (x + 1)`
     will fit a polynomial in $(x - 1) / (x + 1)$;
  * `weight`: specifies the function used to weight the error;
  * `dtype`: sets the data type used in calculations;
  * `trace`: shows the error plot resulting after each iteration.

  Note that options `relative` and `weigth` cannot be specified simultaneously, as both `odd` and `even` options.

  If any of options `odd` or `even` is specified and `degree_powers` is an integer (specifying the degree of the polynomial to fit), then the fitted polynomial will be also odd or even, respectively.

  `p, prec = remezfit(f, a, b, degree_powers, [, option...])`  returns in `p` the fitted polynomial coefficients —starting from the lowest order coefficient, thus suitable to be used by `remezfit.hornersparse()` below—. The `prec` returned value is:

  * the maximum weighted absolute error, when option `relative` is `False`
  * the minimum number of correct digits, otherwise

  Usage examples of `remezfit.remezfit()`:

  * `p, prec = remezfit(lambda x: np.sin(x), 0, np.pi/2, 5, relative = True, odd = True, dtype = np.single, trace = True)`

    will fit an odd, 5th degree, polynomial in $x$ to $\sin(x)$ along interval (0, $\pi/2$) minimizing maximum relative error. Will use single precision numbers during calculations and will show the error plot after each iteration. `p` will contain the polynomial coefficients and `prec` the number of correct digits of the approximation.

  * `p, prec = remezfit(lambda x: (np.cos(x) - 1)/np.power(x, 2), 1e-15, np.pi/2, [0, 2, 4], even = True, weight = lambda x: np.power(x, 2))`

    will fit a polynomial in $x$ with with powers {0, 2, 4} to $(\cos(x) - 1)/x^2$ along interval (1e-15, $\pi/2$) minimizing maximum weighted (by $x^2$) error. `p` will contain the coefficients for powers {0, 2, 4} and `prec` the maximum weighted error of the approximation.

  * `p, prec = remezfit(lambda x: np.log(x), 1, np.sqrt(2), 3, odd = True, arg = lambda x: (x - 1) / (x + 1))`

    will fit an odd, 3rd degree, polynomial in $(x - 1) / (x + 1)$ to $\log(x)$ along interval (1, $\sqrt2$) minimizing the maximum absolute error. `p` will contain the polynomial coefficients and `prec` the maximum absolute error of the approximation.

* Provides function `remezfit.hornersparse()` that evaluates a polynomial using Horner's scheme managing the case of some, or many, coefficients being 0. `hornersparse(x, powers, coeffs)` evaluates at points `x` a polynomial with (integer) powers given in `powers` and coefficients given in `coeffs`. Both `powers` and `coeffs` must have the same number of elements and `powers` must be a strictly increasing sequence. All of `x`, `powers` and `coeffs` can be lists or `numpy` arrays, whereas the return value is always a `numpy` array.

  Usage example:

  * `y = hornersparse(x, [0, 1, 3, 15], [7, 9, 4, 5])`

  will evaluate polynomial $y = 7 + 9x + 4x^3 + 5x^{15}$ at points `x`.

* File `remezfit` can also be used from the command line. 
