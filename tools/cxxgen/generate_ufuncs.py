import sys

from generate_concrete_cfuncs import generate_concrete_cfuncs
from generate_ufunc_extmod import generate_ufunc_extmod


def generate(cxxgenpath):
    sys.path.append(cxxgendefpath)
    from define_cxxgen_extmods import extmods
    sys.path.pop()
    del sys.modules['define_cxxgen_extmods']
    for extmod in extmods:
        modulename = extmod.modulename
        for header, funclist in extmod.funcs.items():
            generate_concrete_cfuncs(cxxgenpath, header, funclist)
            generate_ufunc_extmod(modulename, cxxgenpath, header, funclist)


if __name__ == "__main__":
    import argparse
    import pathlib
    from os import path

    descr = 'Generate ufuncs from C++ templated functions'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', type=pathlib.Path)
    args = parser.parse_args()

    cxxgendefpath = path.join(args.path)
    generate(cxxgendefpath)
