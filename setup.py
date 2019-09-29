from os.path import join


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('npuff')
    config.add_extension('npuff._logfact',
                         extra_compile_args=['-std=c99'],
                         sources=[join('src', 'logfactorial', 'logfactorial.c'),
                                  join('src', 'logfactorial', 'logfactorial_ufunc.c')])
    config.add_extension('npuff._peaktopeak',
                         extra_compile_args=['-std=c99'],
                         sources=[join('src', 'peaktopeak', 'peaktopeak_gufunc.c.src')])
    config.add_extension('npuff._ufunc_inspector',
                         extra_compile_args=['-std=c99'],
                         sources=[join('src', 'ufunc-inspector', 'ufunc_inspector.c')])
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='npuff',
          version='0.0.1',
          configuration=configuration)
