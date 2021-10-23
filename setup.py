from os.path import join


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    compile_args = ['-std=c99', '-Werror']
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('ufunclab')
    config.add_subpackage('ufunclab/tests')
    config.add_extension('ufunclab._logfact',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'logfactorial',
                                       'logfactorial.c'),
                                  join('src', 'logfactorial',
                                       'logfactorial_ufunc.c')])
    config.add_extension('ufunclab._issnan',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'issnan',
                                       'issnan_ufunc.c.src')])
    config.add_extension('ufunclab._cross',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'cross',
                                       'cross_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._peaktopeak',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'peaktopeak',
                                       'peaktopeak_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._findfirst',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'findfirst',
                                       'findfirst_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._searchsorted',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'searchsorted',
                                       'searchsorted_gufunc.c.src')])
    config.add_extension('ufunclab._minmax',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'minmax',
                                       'minmax_gufunc.c.src')])
    config.add_extension('ufunclab._means',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'means', 'means_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._meanvar',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'meanvar',
                                       'meanvar_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._mad',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'mad', 'mad_gufunc.c.src')])
    config.add_extension('ufunclab._vnorm',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'vnorm',
                                       'vnorm_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._backlash',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'backlash',
                                       'backlash_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._deadzone',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'deadzone',
                                       'deadzone_ufunc.c.src')])
    config.add_extension('ufunclab._hysteresis_relay',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'hysteresis_relay',
                                       'hysteresis_relay_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._all_same',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'all_same',
                                       'all_same_gufunc.c.src')],
                         include_dirs=[join('src', 'util')])
    config.add_extension('ufunclab._gendot',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'gendot',
                                       'gendotmodule.c')])
    config.add_extension('ufunclab._ufunc_inspector',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'ufunc-inspector',
                                       'ufunc_inspector.c')])
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='ufunclab',
          version='0.0.4.dev1',
          configuration=configuration)
