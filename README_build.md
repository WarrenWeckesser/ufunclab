`ufunclab` uses meson-python (and therefore meson and ninja) as its
build system.

To install the source with the git repository checked out, it should
be sufficient to run

    $ pip install .

To develop locally, I've been using this in the checked out
repository:

    $ meson setup build --prefix $(pwd)/build-install
    $ cd build
    $ ninja
    $ ninja install

That will install the package in the build-install directory.

Note: I'm not a meson expert. In fact, I'm probably Using It Wrong.
If you see incorrect or inefficient uses, create an issue on github
and let me know.
