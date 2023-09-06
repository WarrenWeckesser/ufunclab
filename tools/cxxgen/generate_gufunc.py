import os
import sys
from gufuncgen import gen


def generate_gufunc(cxxgenpath, destdir):
    sys.path.append(cxxgendefpath)
    from define_cxx_gufunc_extmod import extmod
    sys.path.pop()
    del sys.modules['define_cxx_gufunc_extmod']

    # module_name = os.path.join(cxxgenpath, extmod.module + '.cxx')
    module_name = os.path.join(destdir, extmod.module + '.cxx')
    text, headers = gen(extmod, cxxgenpath)
    with open(module_name, 'w') as f:
        f.write(text)
    for header_file, content in headers.items():
        with open(os.path.join(destdir, header_file), 'w') as f:
            f.write(content)


if __name__ == "__main__":
    import argparse
    import pathlib
    from os import path

    descr = ('Generate an extension module that defines a gufunc \n'
             'from C++ templated functions.')
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', type=pathlib.Path,
                        help='Source directory')
    parser.add_argument('destdir', type=pathlib.Path,
                        help='Destination directory')
    args = parser.parse_args()

    cxxgendefpath = path.join(args.path)
    with open(f'/home/warren/mylog.txt', 'a') as f:
        f.write(cxxgendefpath)
        f.write('\n')
    generate_gufunc(cxxgendefpath, args.destdir)
