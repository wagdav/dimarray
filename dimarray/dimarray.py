import numpy as np

class DimArray(np.ndarray):
    """
    >>> dims = [('a', [0,1,2,3,4,5]), ('b', [0,1,2])]
    >>> a = DimArray(np.arange(6*3).reshape((6,3)), dims=dims)
    >>> a._context == None
    True
    >>> b = a[1:3,0::2]
    >>> b._context == None
    True

    >>> a._context == None # Context after creating b must be None
    True

    >>> a.dims == (('a', [0, 1, 2, 3, 4, 5]), ('b', [0, 1, 2]))
    True
    >>> b.dims == (('a', [1, 2]), ('b', [0, 2]))
    True

    >>> c = b[:,0]
    >>> c.dims  == (('a', [1, 2]), ('b', 0))
    True
    """
    def __new__(cls, input_array, dims=None):
        obj = np.asarray(input_array).view(cls)

        obj.dims = tuple(dims)
        obj._context = None

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        self.dims = getattr(obj, 'dims', None)
        context = getattr(obj, '_context', None)
        if context:
            self.dims = context
            self._context = None
            obj._context = None

    def __array_wrap__(self, out_arr, context=None):
        out_arr.dims = getattr(self, 'dims', None)
        out_arr._context = None
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, y):
        self._context=_dims_getitem(self.dims, y)
        return np.ndarray.__getitem__(self, y)


def _dims_getitem(dimensions, key):
    """
    One dimensional tests:
    >>> dims = (('a', [0, 1, 2, 3, 4, 5]),)
    >>> d=_dims_getitem(dims,4)
    >>> d ==  (('a', [4]),)
    True

    >>> d=_dims_getitem(dims, slice(0,3,None))
    >>> d == (('a', [0, 1, 2]),)
    True
    """
    if isinstance(key, int):
        key = (slice(key, key+1),)

    if isinstance(key, slice):
        key = (key,)

    from itertools import izip_longest
    ret=[]
    for islice, idim in izip_longest(key, dimensions,
            fillvalue=slice(None)):
        if np.isscalar(idim[1]):
            ret.append((idim[0], idim[1]))
        else:
            ret.append((idim[0], idim[1][islice]))

    return tuple(ret)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
