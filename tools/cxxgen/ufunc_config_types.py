
from typing import List
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
