
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Func:
    cxxname: str
    ufuncname: str
    types: List[str]
    docstring: str


@dataclass
class ExtMod:
    modulename: str
    funcs: List[Func]


@dataclass
class GUFuncExtMod:
    module: str
    ufuncname: str
    docstring: str
    signature: str
    corefile: str
    corefuncs: Dict[str, List[str]]
    header: str
    nonzero_coredims: List[str] = None
