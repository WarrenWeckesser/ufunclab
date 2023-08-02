import sys
import os
from os.path import join


def get_version():
    """
    Find the value assigned to __version__ in ufunclab/__init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in __init__.py.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open(join("ufunclab", "__init__.py"), "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


def generate_cxxgen_code(dirnames):
    import subprocess

    cwd = os.getcwd()
    os.chdir(join(cwd, 'tools', 'cxxgen'))

    for dirname in dirnames:
        srcpath = join(cwd, 'src', dirname)
        cmd = [sys.executable, 'generate_ufuncs.py', srcpath]
        subprocess.run(cmd)

    os.chdir(cwd)


def generate_cxx_gufunc_extmods(dirnames):
    import subprocess

    cwd = os.getcwd()
    os.chdir(join(cwd, 'tools', 'cxxgen'))

    for dirname in dirnames:
        srcpath = join(cwd, 'src', dirname)
        cmd = [sys.executable, 'generate_gufunc.py', srcpath]
        subprocess.run(cmd)

    os.chdir(cwd)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info

    compile_args = ['-std=c99', '-Werror']
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('ufunclab')
    config.add_subpackage('ufunclab/tests')

    util_include_dir = join('src', 'util')
    npymath_info = get_info("npymath")
    npymath_info['include_dirs'].append(util_include_dir)

    config.add_extension('ufunclab._logfact',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'logfactorial',
                                       'logfactorial.c'),
                                  join('src', 'logfactorial',
                                       'logfactorial_ufunc.c')])

    config.add_extension('ufunclab._loggamma1p',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'loggamma1p',
                                       'loggamma1p.c'),
                                  join('src', 'loggamma1p',
                                       'loggamma1p_ufunc.c')])

    config.add_extension('ufunclab._issnan',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'issnan',
                                       'issnan_ufunc.c.src')])

    _as_srcs = ['abs_squared_concrete.cxx', '_abs_squaredmodule.cxx']
    config.add_extension('ufunclab._abs_squared',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'abs_squared', 'generated', name)
                                  for name in _as_srcs],
                         **npymath_info)

    config.add_extension('ufunclab._cabssq',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'cabssq',
                                       'cabssq_ufunc.c')])

    config.add_extension('ufunclab._log1p',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'log1p',
                                       'log1p_ufunc.c')])

    _ei_srcs = ['expint1_concrete.cxx', '_expint1module.cxx']
    config.add_extension('ufunclab._expint1',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'expint1', 'generated', name)
                                  for name in _ei_srcs],
                         **npymath_info)

    _p1_srcs = ['pow1pm1_concrete.cxx', '_pow1pm1module.cxx']
    config.add_extension('ufunclab._pow1pm1',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'pow1pm1', 'generated', name)
                                  for name in _p1_srcs],
                         **npymath_info)

    _le_srcs = ['logistic_concrete.cxx', '_logisticmodule.cxx']
    config.add_extension('ufunclab._logistic',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'logistic', 'generated', name)
                                  for name in _le_srcs],
                         **npymath_info)

    _ramp_srcs = ['ramp_concrete.cxx', '_rampmodule.cxx']
    config.add_extension('ufunclab._ramp',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'ramp', 'generated', name)
                                  for name in _ramp_srcs],
                         **npymath_info)

    _yj_srcs = ['yeo_johnson_concrete.cxx', '_yeo_johnsonmodule.cxx']
    config.add_extension('ufunclab._yeo_johnson',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'yeo_johnson', 'generated', name)
                                  for name in _yj_srcs],
                         **npymath_info)

    _nm_srcs = ['normal_concrete.cxx', 'erfcx_funcs_concrete.cxx',
                'normalmodule.cxx']
    config.add_extension('ufunclab.normal',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'normal', 'generated', name)
                                  for name in _nm_srcs] +
                                 [join('src', 'normal', 'erfcx_funcs.cxx')],
                         **npymath_info)

    config.add_extension('ufunclab._cross',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'cross',
                                       'cross_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    config.add_extension('ufunclab._peaktopeak',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'peaktopeak',
                                       'peaktopeak_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    config.add_extension('ufunclab._first',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'first',
                                       'first_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    config.add_extension('ufunclab._searchsorted',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'searchsorted',
                                       'searchsorted_gufunc.c.src')],
                         **npymath_info)

    config.add_extension('ufunclab._means',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'means', 'means_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    _meanvar_srcs = ['meanvar_gufunc.h', '_meanvarmodule.cxx']
    config.add_extension('ufunclab._meanvar',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'meanvar', name)
                                  for name in _meanvar_srcs],
                         **npymath_info)

    _corr_srcs = ['corr_gufunc.h', '_corrmodule.cxx']
    config.add_extension('ufunclab._corr',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'corr', name)
                                  for name in _corr_srcs],
                         **npymath_info)

    _mad_srcs = ['mad_gufunc.h', '_madmodule.cxx']
    config.add_extension('ufunclab._mad',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'mad', name)
                                  for name in _mad_srcs],
                         **npymath_info)

    _vnorm_srcs = ['vnorm_gufunc.h', '_vnormmodule.cxx']
    config.add_extension('ufunclab._vnorm',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'vnorm', name)
                                  for name in _vnorm_srcs],
                         **npymath_info)

    _tri_area_srcs = ['tri_area_gufunc.h', '_tri_areamodule.cxx']
    config.add_extension('ufunclab._tri_area',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'tri_area', name)
                                  for name in _tri_area_srcs],
                         **npymath_info)

    config.add_extension('ufunclab._backlash',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'backlash',
                                       'backlash_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    config.add_extension('ufunclab._fillnan1d',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'fillnan1d',
                                       'fillnan1d_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    config.add_extension('ufunclab._linear_interp1d',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'linear_interp1d',
                                       'linear_interp1d_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    _dz_srcs = ['deadzone_concrete.cxx', '_deadzonemodule.cxx']
    config.add_extension('ufunclab._deadzone',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'deadzone', 'generated', name)
                                  for name in _dz_srcs],
                         **npymath_info)

    _semivar_srcs = ['semivar_concrete.cxx', 'semivarmodule.cxx']
    config.add_extension('ufunclab.semivar',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'semivar', 'generated', name)
                                  for name in _semivar_srcs],
                         **npymath_info)

    _tp_srcs = ['trapezoid_pulse_concrete.cxx', '_trapezoid_pulsemodule.cxx']
    config.add_extension('ufunclab._trapezoid_pulse',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'trapezoid_pulse', 'generated',
                                       name)
                                  for name in _tp_srcs],
                         **npymath_info)

    config.add_extension('ufunclab._hysteresis_relay',
                         extra_compile_args=compile_args,
                         sources=[join('src', 'hysteresis_relay',
                                       'hysteresis_relay_gufunc.c.src')],
                         include_dirs=[util_include_dir])

    _all_same_srcs = ['all_same_gufunc.h', '_all_samemodule.cxx']
    config.add_extension('ufunclab._all_same',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'all_same', name)
                                  for name in _all_same_srcs],
                         **npymath_info)

    _minmax_srcs = ['minmax_gufunc.h', '_minmaxmodule.cxx']
    config.add_extension('ufunclab._minmax',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'minmax', name)
                                  for name in _minmax_srcs],
                         **npymath_info)

    _mvlogbeta_srcs = ['multivariate_logbeta_gufunc.h',
                       '_multivariate_logbetamodule.cxx']
    config.add_extension('ufunclab._multivariate_logbeta',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'multivariate_logbeta', name)
                                  for name in _mvlogbeta_srcs],
                         **npymath_info)

    _step_srcs = ['step_funcs_concrete.cxx', '_stepmodule.cxx']
    config.add_extension('ufunclab._step',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'step', 'generated', name)
                                  for name in _step_srcs],
                         **npymath_info)

    _next_srcs = ['next_funcs_concrete.cxx', '_nextmodule.cxx']
    config.add_extension('ufunclab._next',
                         extra_compile_args=['-std=c++11', '-Werror'],
                         sources=[join('src', 'next', 'generated', name)
                                  for name in _next_srcs],
                         **npymath_info)

    _sosfilter_srcs = ['sosfilter_gufunc.h', '_sosfiltermodule.cxx']
    config.add_extension('ufunclab._sosfilter',
                         extra_compile_args=['-std=c++11', '-Werror', '-O3'],
                         sources=[join('src', 'sosfilter', name)
                                  for name in _sosfilter_srcs],
                         **npymath_info)

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

    generate_cxxgen_code(['abs_squared', 'deadzone', 'expint1',
                          'ramp', 'logistic', 'next', 'normal', 'pow1pm1',
                          'semivar', 'step', 'trapezoid_pulse',
                          'yeo_johnson'])

    generate_cxx_gufunc_extmods(['all_same', 'corr', 'mad', 'meanvar',
                                 'minmax', 'multivariate_logbeta',
                                 'sosfilter', 'tri_area', 'vnorm'])

    setup(name='ufunclab',
          version=get_version(),
          configuration=configuration)
