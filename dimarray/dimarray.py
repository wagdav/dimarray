"""
1D case
-------

>>> a = DimArray(range(10), dims=[('t', range(10))])
>>> a.dims
OrderedDict([('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])
>>> a[:5].dims
OrderedDict([('t', [0, 1, 2, 3, 4])])


2D case
-------

>>> import numpy as np
>>> dims = [('a', range(6)), ('b', range(3))]
>>> a = DimArray(np.arange(6*3).reshape((6,3)), dims=dims)
>>> a.dims
OrderedDict([('a', [0, 1, 2, 3, 4, 5]), ('b', [0, 1, 2])])
>>> b = a[1:3, 0::2]
>>> b.dims
OrderedDict([('a', [1, 2]), ('b', [0, 2])])
>>> b[:, 0].dims
OrderedDict([('a', [1, 2]), ('b', 0)])
>>> b[0, :].dims
OrderedDict([('a', 1), ('b', [0, 2])])


Slice of a slice
----------------

>>> a = DimArray(np.random.rand(6,3), [('X', range(6)), ('comp', list('xyz'))])
>>> b = a[0,:]
>>> b.dims
OrderedDict([('X', 0), ('comp', ['x', 'y', 'z'])])
>>> b[0:2].dims
OrderedDict([('X', 0), ('comp', ['x', 'y'])])


Slices with Ellipsis
--------------------

>>> dims = [('a', range(2)), ('b', range(3)), ('c', range(4)), ('d', range(5))]
>>> a = DimArray(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dims=dims)
>>> a[0, ..., 0].dims
OrderedDict([('a', 0), ('b', [0, 1, 2]), ('c', [0, 1, 2, 3]), ('d', 0)])
>>> a[0, ...].dims
OrderedDict([('a', 0), ('b', [0, 1, 2]), ('c', [0, 1, 2, 3]), ('d', [0, 1, 2, 3, 4])])
>>> a[..., 0].dims
OrderedDict([('a', [0, 1]), ('b', [0, 1, 2]), ('c', [0, 1, 2, 3]), ('d', 0)])


Newaxis
-------

>>> import numpy as np
>>> a = DimArray(range(10), dims=[('t', range(10))])
>>> a[np.newaxis].dims
OrderedDict([('newaxis0', [None]), ('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])
>>> b = a[..., np.newaxis]
>>> b.dims
OrderedDict([('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('newaxis1', [None])])
>>> b[:, 0].dims
OrderedDict([('t', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('newaxis1', None)])


Non-managed ranges
------------------

>>> dims = [('a', range(2)), ('b', range(3)), ('c', [None]), ('d', range(5))]
>>> a = DimArray(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dims=dims)
>>> a[0, ..., 0].dims
OrderedDict([('a', 0), ('b', [0, 1, 2]), ('c', [None]), ('d', 0)])
>>> a[..., 0, 0].dims
OrderedDict([('a', [0, 1]), ('b', [0, 1, 2]), ('c', None), ('d', 0)])


"Hardcore" tests
----------------
>>> dims = [('a', range(2)), ('b', range(3)), ('c', [None]), ('d', range(5))]
>>> a = DimArray(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dims=dims)
>>> a[0:1, ..., np.newaxis, 0, np.newaxis].dims
OrderedDict([('a', [0]), ('b', [0, 1, 2]), ('c', [None]), ('newaxis0', [None]), ('d', 0), ('newaxis1', [None])])


Numpy ufuncs that do not alter the shape
----------------------------------------

>>> import numpy as np
>>> dims = [('a', range(2)), ('b', range(3)), ('c', [None]), ('d', range(5))]
>>> a = DimArray(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dims=dims)
>>> b = np.exp(a)
>>> b.dims == a.dims
True
"""

from itertools import izip_longest
import numpy as np

from sys import version_info
if version_info < (2, 7):
    from odict import OrderedDict
else:
    from collections import OrderedDict

class DimArray(np.ndarray):
    """
    TODO
    """
    _counter = 0
    empty_dim_range = [None]

    def __new__(cls, input_array, dims):
        obj = np.asarray(input_array).view(cls)
        obj.dims = OrderedDict(dims)
        try:
            obj._check_dims()
        except ValueError:
            raise ValueError('invalid dimension info given.')
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        if isinstance(obj, DimArray):
            if obj._dims_buffer is not None:
                self.dims = obj._dims_buffer
                obj._dims_buffer = None
            else:
                self.dims = obj.dims
            try:
                self._check_dims()
            except ValueError:
                raise NotImplementedError('invalid dimension info (unsupported '
                    'NumPy function called?)')
        else:
            self.dims = None
        self._dims_buffer = None

    #def __array_wrap__(self, out_arr, context=None):
    #    out_arr.dims = getattr(self, 'dims', None)
    #    return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, key):
        self._dims_buffer = self._dims_getitem(key)
        item = np.ndarray.__getitem__(self, key)
        if not np.isscalar(item) and item.size == 0:
            raise ValueError('zero size arrays are not allowed.')
        return item

    # Deprecated, see http://docs.python.org/library/operator.html
    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def __setitem__(self, key, value):
        raise NotImplementedError

    # Deprecated, see http://docs.python.org/library/operator.html
    def __setslice__(self, start, stop, value):
        raise NotImplementedError

    def iter_dims_not_singleton(self):
        for name, range_ in self.dims.iteritems():
            if isinstance(range_, (tuple, list)):
                yield name, range_

    def iter_dims(self):
        for dim in self.dims.iteritems():
            # See PEP 342 for the new yield expression
            # This allows to temporarily insert a user defined element into the
            # iteration, exploited when handling newaxis.
            value_sent = (yield dim)
            if value_sent is not None:
                # send() immediately yields a value which we discard
                yield None
                # Next time we will yield the value sent
                yield value_sent

    @property
    def dims_not_singleton(self):
        return OrderedDict([dim for dim in self.iter_dims_not_singleton()])

    def _check_dims(self):
        dims = self.dims_not_singleton
        if len(self.shape) != len(dims):
            raise ValueError
        for length, range_ in zip(self.shape, dims.itervalues()):
            if range_ != self.empty_dim_range and length != len(range_):
                raise ValueError

    def _dims_getitem(self, key):
        """
        TODO: new doctests
        """
        if isinstance(key, (int, slice)):
            key = (key,)
        elif key is Ellipsis:
            key = (slice(None),)
        elif key is np.newaxis:
            key = (np.newaxis, slice(None))

        key = list(key)
        ret = []
        iter_dims = self.iter_dims()

        i, ipad = 0, 0
        for rng, dim in izip_longest(key, iter_dims, fillvalue=slice(None)):
            if rng is np.newaxis:
                # Repeat the dimension unless newaxis is at the end
                if dim != slice(None):
                    iter_dims.send(dim)
                ret.append(('newaxis' + str(self._counter),
                           self.empty_dim_range))
                self._counter += 1
                continue

            # Add all existing singleton dimensions of dims to the returned
            # dims, too.
            while not isinstance(dim[1], (tuple, list)):
                ret.append(dim)
                try:
                    # Skip their processing
                    dim = iter_dims.next()
                except StopIteration:
                    # If last dimension is a singleton return immediately
                    if not isinstance(dim[1], (tuple, list)):
                        return OrderedDict(ret)
                    else: # If no, let it be processed
                        break

            # If an ellipsis is present set a counter to insert as many full
            # slices instead as necessary. N.B. newaxis in key does not count!
            if rng is Ellipsis:
                ipad = len(self.shape) - (len(key) - key.count(np.newaxis))
                rng = slice(None)

            if ipad > 0:
                key.insert(i + 1, slice(None))
                ipad -= 1

            dim_name, dim_range = dim

            # Only slice dimensions for which the range is managed, too
            if dim_range != self.empty_dim_range:
                new_range = dim_range[rng]
            # However, take care if a name-only dimension is set to singleton
            elif isinstance(rng, int):
                new_range = None
            else:
                new_range = dim_range
            ret.append((dim_name, new_range))
            i += 1
        return OrderedDict(ret)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
