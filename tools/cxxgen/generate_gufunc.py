import os
import sys
from gufuncgen import gen


def generate_gufunc(cxxgenpath):
    sys.path.append(cxxgendefpath)
    from define_cxx_gufunc_extmod import extmod
    sys.path.pop()
    del sys.modules['define_cxx_gufunc_extmod']

    module_name = os.path.join(cxxgenpath, extmod.module + 'module.cxx')
    text = gen(name=extmod.ufuncname,
               # varnames=varnames,
               signature=extmod.signature,
               corefuncs=extmod.corefuncs,
               docstring=extmod.docstring,
               header=extmod.header)
    with open(module_name, 'w') as f:
        f.write(text)
    f.close()


if __name__ == "__main__":
    import argparse
    import pathlib
    from os import path

    descr = ('Generate an extension module that defines a gufunc \n'
             'from C++ templated functions.')
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', type=pathlib.Path)
    args = parser.parse_args()

    cxxgendefpath = path.join(args.path)
    generate_gufunc(cxxgendefpath)
