import os
import sys
from gufuncgen import gen


def generate_gufunc(cxxgenpath):
    print("generate_gufunc: cxxgenpath:", cxxgenpath)
    sys.path.append(cxxgendefpath)
    from define_cxx_gufunc_extmod import extmod
    sys.path.pop()
    print("generate_gufunc: extmod.module:   ", extmod.module)
    print("generate_gufunc: extmod.ufuncname:", extmod.ufuncname)
    print("generate_gufunc: extmod.signature:", extmod.signature)
    print("generate_gufunc: extmod.header:   ", extmod.header)
    del sys.modules['define_cxx_gufunc_extmod']

    module_name = os.path.join(cxxgenpath, extmod.module + 'module.cxx')
    print('generate_gufunc: module_name:', module_name)
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
    print(f"****** Calling generate_gufunc({cxxgendefpath})")
    generate_gufunc(cxxgendefpath)
