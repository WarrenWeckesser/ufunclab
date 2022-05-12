
from typing import List
from dataclasses import dataclass


@dataclass
class UFuncSource:

    # Name of the C++ function
    funcname: str

    # Type signatures handled by the C++ function (as template
    # parameters, if there is more than one type signature).
    typesignatures: List[str]


@dataclass
class UFunc:

    # Name of the ufunc in Python
    name: str

    # C++ header filename that defines the function
    header: str

    # Docstring of the ufunc.
    # The first line of the docstring *must* have the form
    #     ufunc(var1, var2, /, ...)
    # That is, the name of function with the core variables,
    # a forward slash, an ellipsis and the closing parenthesis.
    # This line is read by the ufunc generation code to get the
    # variable names of the ufunc.
    docstring: str

    # The shape signature of the ufunc.  For standard ufuncs,
    # this can be set to None.
    signature: str

    # List of UFuncSource instances that define the core C++ functions
    # that will be wrapped by generated loop functions that are used
    # to implment the ufunc.
    sources: List[UFuncSource]

    # List of core dimension names (i.e. names that appear in the
    # shape signature) that must have a length of at least 1.
    nonzero_coredims: List[str] = None


@dataclass
class UFuncExtMod:

    # Module name in Python
    module: str

    # Module docstring
    docstring: str

    # List of UFunc instance that define the ufuncs to be implemented
    # in the extension module.
    ufuncs: List[UFunc]


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
