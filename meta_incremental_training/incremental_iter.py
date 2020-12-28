class IncrementalIterator:
    """ Class for incrementally iterating over the dataset, which works with other user-defined class/function.
    Attributes:
        - data_pool: dict with sequence (e.g. list) of data
    """

    def __init__(self, window_size: int = -1, is_cuda: bool = True):
        """
        :param window_size: size per window
        :param is_cuda: boolean, true converting data to cuda.tensor
        """
        self.data_pool = None
        self.windows = []
        self.is_cuda = is_cuda
        self.size_per_win = window_size

    def iter_from_data(self, index: int, window_range: int, process_function):
        """
        Returns the iterator containing windows between [i-w, i-1] or returns the i-th window during i-th iteration.
        This method directly get windows from data_pool (original data).
        :param window_range: w, None or 0 means return only the i-th window. If it's negative, it will iterator over
        [0, i-1]-th windows
        :param index: i
        :param process_function: function object, to process the batch data before input to the model
        """
        if not window_range or window_range == 0:
            start_index = index * self.size_per_win
            end_index = start_index + self.size_per_win
            return process_function(self.data_pool[start_index: end_index])
        elif window_range == -1:
            window_range = index
        else:
            raise ValueError("window_range should be non-negative integer or -1.")

        for i in range(max(index - window_range, 0), index - 1):
            start_index = i * self.size_per_win
            end_index = i + self.size_per_win
            yield process_function(self.data_pool[start_index: end_index])

    def iter_from_list(self, index: int, window_range: int):
        """
            Returns the iterator containing windows between [i-w, i-1] or returns the i-th window during i-th iteration.
        This method applies on self.windows.
        :param window_range: w, None or 0 means return only the i-th window. If it's negative, it will iterator over
        [0, i]-th windows
        :param index: i
        """
        if not window_range or window_range == 0:
            return self.windows[index]  # to return an iterable
        elif window_range == -1:
            window_range = index
        else:
            raise ValueError("window_range should be non-negative integer or -1.")

        for i in range(index, max(index - window_range, 0)-1, -1):
            yield self.windows[i]
