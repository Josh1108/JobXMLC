from functools import wraps

DATA_FILTER_REGISTRY = {}
ENCODER_REGISTRY = {}
ENCODER = "encoder"
DATA_FILTER = "data_filter"


def optional_params(func):
    """Allow a decorator to be called without parentheses if no kwargs are given.

    parameterize is a decorator, function is also a decorator.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        """If a decorator is called with only the wrapping function just execute the real decorator.
           Otherwise return a lambda that has the args and kwargs partially applied and read to take a function as an
           argument.

        *args, **kwargs are the arguments that the decorator we are parameterizing is called with.

        the first argument of *args is the actual function that will be wrapped
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)

    return wrapped


@optional_params
def register(cls, _type: str, _name: str):
    if _type == ENCODER:
        ENCODER_REGISTRY[_name] = cls
    elif _type == DATA_FILTER:
        DATA_FILTER_REGISTRY[_name] = cls
    else:
        raise RuntimeError(f"No suitable registry found for type {_type}")
    return cls