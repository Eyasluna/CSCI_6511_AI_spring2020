# _*_coding:utf-8 _*_
#author: Yibo Fu
#G25190736

import time
import functools


def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        print(f"Start [{method.__name__}]:")
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print(f"End [{method.__name__}].  Time elapsed:{end_time - start_time: 0.3f} sec. ")
        return result
    return timed