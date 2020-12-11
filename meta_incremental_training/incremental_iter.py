class IncrementalIterator:
    """ Class for incrementally iterating over the dataset, which works with other user-defined class/function. """
    def __init__(self, data_pool, window_size: int = 1024, is_cuda: bool = True):
        """:param data_pool: dict with sequence (e.g. list) of data
        :param window_size: size per window
        :param is_cuda: boolean, true converting data to cuda.tensor """
        self.data_pool = data_pool
        self.is_cuda = is_cuda
        self.size_per_win = window_size

    def iter(self, index: int, window_range: int, process_function):
        """
        Returns the iterator containing windows between [i-w, i-1] or returns the i-th window during i-th iteration.
        :param window_range: w, None or 0 means return only the i-th window, or it must be positive integer
        :param index: i
        :param process_function: function object, to process the batch data before input to the model
        """
        if not window_range or window_range == 0:
            start_index = index * self.size_per_win
            end_index = start_index + self.size_per_win
            return process_function(self.data_pool[start_index: end_index])
        elif window_range < 0:
            raise ValueError("window_range should be non-negative integer.")

        for i in range(max(index-window_range, 0), index-1):
            start_index = i * self.size_per_win
            end_index = i + self.size_per_win
            yield process_function(self.data_pool[start_index: end_index])
