'''
Author: Liu Xin
Date: 2021-11-16 11:06:01
LastEditors: Liu Xin
LastEditTime: 2021-11-20 17:36:47
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
if __name__ == '__main__':
    a = None
    add(*a)