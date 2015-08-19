from warnings import catch_warnings, filterwarnings

# TODO global to disable this?
def ignore_future_warnings(func):
    """
    Decorator to suppress future warnings
    """
    def suppressed_fuction(*args, **kwargs):
        with catch_warnings():
            filterwarnings("ignore", category=FutureWarning)
            return func(*args, **kwargs)

    suppressed_fuction.__name__ = func.__name__
    suppressed_fuction.__doc__ = func.__doc__
    suppressed_fuction.__dict__.update(func.__dict__)
    return suppressed_fuction


class ConfigurationException(Exception):
    """
    Raised when an instance of a class hasn't been properly configured
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class OutputUnallocatedException(Exception):
    """
    Raised when the output of an ImageChunkerWithOutput is requested but has not yet been allocated
    """
    
    def __init__(self, message=None):
        if message is not None:
            self.message = message
        else:
            self.message = "ImageChunkerWithOutput output has not yet been allocated!"

    def __str__(self):
        return self.message
