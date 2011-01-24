"""
1D case
-------

>>> a = DimArray(range(10),dims=[('t',range(10))])
>>> a.dims == (('t', range(10)),)
True

>>> a[:5].dims == (('t', [0, 1, 2, 3, 4]),)
True


2D case
-------

>>> dims = [('a', [0,1,2,3,4,5]), ('b', [0,1,2])]
>>> a = DimArray(np.arange(6*3).reshape((6,3)), dims=dims)
>>> b = a[1:3,0::2]
>>> a.dims == (('a', [0, 1, 2, 3, 4, 5]), ('b', [0, 1, 2]))
True
>>> b.dims == (('a', [1, 2]), ('b', [0, 2]))
True
>>> c = b[:,0]
>>> c.dims  == (('a', [1, 2]), ('b', 0))
True


Slices with Ellipsis
--------------------

>>> dims = [('a', '12'), ('b', '123'), ('c', '1234'), ('d', '12345')]
>>> a = DimArray(np.arange(2*3*4*5).reshape((2,3,4,5)), dims=dims)
>>> a[0,...,0].shape
(3, 4)

>>> a[0,...,0].dims
(('a', '1'), ('b', '123'), ('c', '1234'), ('d', '1'))


Newaxis
-------

>>> import numpy as np
>>> a = DimArray(range(10),dims=[('t',range(10))])
>>> a.dims == (('t', range(10)),)
True
>>> a[...,np.newaxis].dims
(('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('', [None]))

>>> a[np.newaxis].dims
(('', [None]), ('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


>>> b = a[...,np.newaxis]
>>> b[:,0].dims
(('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('', None))
"""

from itertools import izip_longest
import numpy as np

class DimArray(np.ndarray):
    """
    >>> dims = [('a', [0,1,2,3,4,5]), ('b', [0,1,2])]
    >>> a = DimArray(np.arange(6*3).reshape((6,3)), dims=dims)
    >>> b = a[1:3,0::2]
    """
    def __new__(cls, input_array, dims=None):
        obj = np.asarray(input_array).view(cls)
        obj.dims = tuple(dims)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        if not hasattr(self, 'dims'):
            self.dims = getattr(obj, 'dims', None)

    def __array_wrap__(self, out_arr, context=None):
        out_arr.dims = getattr(self, 'dims', None)
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, y):
        out_arr = np.ndarray.__getitem__(self, y)
        if not np.isscalar(out_arr):
            out_arr.dims = _dims_getitem(self.dims, y)
        return out_arr

    def __getslice__(self, i, j):
        out_arr = np.ndarray.__getslice__(self, i, j)
        if not np.isscalar(out_arr):
            out_arr.dims = _dims_getitem(self.dims, slice(i,j))
        return out_arr

def _dims_getitem(dimensions, key):
    """
    One dimensional tests:
    >>> dims = (('a', [0, 1, 2, 3, 4, 5]),)
    >>> d=_dims_getitem(dims,4)
    >>> d ==  (('a', [4]),)
    True

    >>> d=_dims_getitem(dims, slice(0, 3, None))
    >>> d == (('a', [0, 1, 2]),)
    True

    >>> d=_dims_getitem(dims, slice(None, 3,None))
    >>> d == (('a', [0, 1, 2]),)
    True

    >>> d=_dims_getitem(dims, slice(3, None, None))
    >>> d == (('a', [3, 4, 5]),)
    True
    """
    if isinstance(key, int):
        key = (slice(key, key+1),)

    if isinstance(key, slice):
        key = (key,)

    if key is Ellipsis:
        key = (slice(None),)

    if key is None:
        key = (None,slice(None))

    empty_dimension = ('',[None])

    key,dimensions = list(key), list(dimensions)
    ret = []
    ipad, i = 0, 0
    for rng, dim in izip_longest(key, dimensions,
                            fillvalue=slice(None)):
        if rng is Ellipsis:
            ipad = len(dimensions) - len(key)
            rng = slice(None)

        if ipad > 0:
            key.insert(i+1, slice(None))
            ipad -= 1

        if rng is None:
            dimensions.insert(i+1, dim)
            ret.append(empty_dimension)
        else:
            dim_name, dim_range = dim
            new_range = dim_range[rng]
            ret.append((dim_name, new_range))

        i += 1

    return tuple(ret)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
