
import functools

def trace_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n[TRACE] Call {func.__module__}.{func.__name__}")
        for i, arg in enumerate(args):
            t = type(arg)
            try:
                info = f"{t} shape={getattr(arg,'shape',None)} len={getattr(arg,'__len__',lambda:None)()}"
            except Exception:
                info = str(t)
            print(f"  arg[{i}]: {info}")
        for k, v in kwargs.items():
            print(f"  kwarg {k}: {type(v)}")
        result = func(*args, **kwargs)
        if hasattr(result, "shape"):
            print(f"  [TRACE] {func.__name__} returns shape {result.shape}")
        elif isinstance(result, (tuple, list)) and hasattr(result[0], "shape"):
            print(f"  [TRACE] {func.__name__} returns shape {result[0].shape}")
        else:
            print(f"  [TRACE] {func.__name__} returns {type(result)}")
        return result
    return wrapper


