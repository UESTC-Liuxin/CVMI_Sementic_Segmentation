'''
Author: Liu Xin
Date: 2021-11-16 11:06:01
LastEditors: Liu Xin
LastEditTime: 2021-11-25 11:51:24
Description: file content
FilePath: /CVMI_Sementic_Segmentation/tools/debug.py
'''


def logger(func):
    print(func)

    def wrapper(*args, **kw):
        print(args)
        print('{} is called...'.format(func.__name__))
        func(*args, **kw)
    return wrapper


@logger
def add(x, y):
    print(x+y)


def add():
    print("ok..")
    
    
from torch.optim.adam import Adam
if __name__ == '__main__':
    print(Adam.__name__)
    from collections import OrderedDict
    A = OrderedDict()
    value = A.get("a", 0)
    print(value)