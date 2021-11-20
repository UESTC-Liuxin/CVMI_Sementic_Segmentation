'''
Author: Liu Xin
Date: 2021-11-16 10:44:06
LastEditors: Liu Xin
LastEditTime: 2021-11-19 21:23:59
Description: regist module by using decorators(detail by:https://blog.csdn.net/qq_20549061/article/details/107883272)
FilePath: /CVMI_Sementic_Segmentation/utils/registry.py
'''


from os import times


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(name={self._name}, ' \
            f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """
        @description  :
        @param  :
        @Returns  :
        """
        return self._module_dict[key]

    def _register_module(self, module_class, module_name=None, force=False):
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls
        return _register


def build(cfg, registry: Registry):
    """
    @description  :
    @param  :
    @Returns  :
    """
    obj_cls = registry.module_dict[cfg["name"]]
    cfg = cfg.copy()
    cfg.pop("name")
    return obj_cls(**cfg)