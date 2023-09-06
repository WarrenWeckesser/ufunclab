import sys
import shutil
from os import path

from generate_concrete_cfuncs import generate_concrete_cfuncs
from generate_ufunc_extmod import generate_ufunc_extmod


def generate(srcdir, destdir):
    sys.path.append(srcdir)
    from define_cxxgen_extmods import extmods
    sys.path.pop()
    del sys.modules['define_cxxgen_extmods']

    for extmod in extmods:
        for header, funclist in extmod.funcs.items():
            generate_concrete_cfuncs(srcdir, header, funclist, destdir)
            shutil.copyfile(path.join(srcdir, header),
                            path.join(destdir, header))
        generate_ufunc_extmod(srcdir, extmod, destdir)
        if extmod.extra_sources is not None:
            for extra_source in extmod.extra_sources:
                shutil.copyfile(path.join(srcdir, extra_source),
                                path.join(destdir, extra_source))


if __name__ == "__main__":
    import argparse
    import pathlib

    descr = 'Generate ufuncs from C++ templated functions'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('src', type=pathlib.Path)
    parser.add_argument('dest', type=pathlib.Path)
    args = parser.parse_args()

    generate(path.join(args.src.resolve()),
             path.join(args.dest.resolve()))
