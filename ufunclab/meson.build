#=========================================================================
# The code from here to the next line of ='s was copied from SciPy
# and edited.
#
# NumPy include directory - needed in all submodules
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
#

print_numpy_include = 'import numpy as np; print(np.get_include())'

incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
    incdir_numpy = run_command(py3,
      '-c',
      print_numpy_include,
      check: true
    ).stdout().strip()

    # We do need an absolute path to feed to `cc.find_library` below
    _incdir_numpy_abs = run_command(py3,
      ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
      check: true
    ).stdout().strip()
else
    _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

npymath_path = _incdir_numpy_abs / '..' / 'lib'
npymath_lib = cc.find_library('npymath', dirs: npymath_path)

#=========================================================================


#----------------------------------------------------------------------
# Extension modules for gufuncs implemented in C++ that use the code
# generator in tools/cxxgen/generate_gufunc.py.
#----------------------------------------------------------------------

gufunc_cxx_src_dirs = [
    'all_same',
    'corr',
    'mad',
    'meanvar',
    'minmax',
    'multivariate_logbeta',
    'sosfilter',
    'tri_area',
    'vnorm'
]

foreach src_dir : gufunc_cxx_src_dirs

    src_name = src_dir + '_module_src'    # e.g. 'all_same_module_src'
    header_name = src_dir + '_gufunc.h'   # e.g. 'all_same_gufunc.h'
    module_name = '_' + src_dir           # e.g. '_all_same'
    output = module_name + '.cxx'         # e.g. '_all_same.cxx'
    pth = join_paths('../src', src_dir)   # e.g. '../src/all_same'

    src = custom_target(src_name,
                        input : ['../tools/cxxgen/generate_gufunc.py',
                                 join_paths(pth, 'define_cxx_gufunc_extmod.py'),
                                 join_paths(pth, header_name)],
                        output : output,
                        command : [py3, '@INPUT0@', pth, './ufunclab'])

    py3.extension_module(
        module_name,
        [src, '../src/util/ufunc_tools.h'],
        include_directories: [inc_np],
        dependencies: [np_dep, npymath_lib],
        install: true,
        subdir: 'ufunclab'
    )

endforeach

#----------------------------------------------------------------------
# C++ extension modules for ufuncs that use the code generation script
# in tools/cxxgen/generate_ufuncs.py.  This meson code takes advantage
# of the simplicity of these particular extension modules; the code for
# the ufuncs in the `normal` subdirectory requires some customization
# and it not include here.
#----------------------------------------------------------------------

ufunc_cxx_src_dirs_simple = [
    'abs_squared',
    'deadzone',
    'expint1',
    'logistic',
    'next',
    # 'normal',
    'pow1pm1',
    'ramp',
    'semivar',
    'step',
    'trapezoid_pulse',
    'yeo_johnson'
]

foreach src_dir : ufunc_cxx_src_dirs_simple

    src_name = src_dir + '_module_src'       # e.g. 'abs_squared_module_src'
    src_header = src_dir + '.h'              # e.g. 'abs_squared.h'
    if src_dir != 'semivar'
        module_name = '_' + src_dir          # e.g. '_abs_squared'
    else
        module_name = src_dir
    endif
    output = module_name + '.cxx'            # e.g. '_abs_squared.cxx'
    src_pth = join_paths('../src', src_dir)  # e.g. '../src/abs_squared'

    # Run tools/cxxgen/generate_ufuncs.py to generate the source code
    # for this extension module.
    mod_src = custom_target(src_name,
                            input : ['../tools/cxxgen/generate_ufuncs.py',
                                     join_paths(src_pth, 'define_cxxgen_extmods.py'),
                                     join_paths(src_pth, src_header)],
                            output : [output,
                                      src_header,
                                      src_dir + '_concrete.h',
                                      src_dir + '_concrete.cxx'],
                            command : [py3, '@INPUT0@', src_pth, './ufunclab'])

    py3.extension_module(
        module_name,
        [mod_src],
        include_directories: [inc_np],
        dependencies: [np_dep, npymath_lib],
        install: true,
        subdir: 'ufunclab'
    )

endforeach


#----------------------------------------------------------------------
# Build for the `normal` extension module.  It uses the script in
# tools/cxxgen/generate_ufuncs.py, but it has more source files,
# so it doesn't fit the simple case used above.
#----------------------------------------------------------------------

mod_src = custom_target('normal_module_src',
                        input : ['../tools/cxxgen/generate_ufuncs.py', 
                                 '../src/normal/define_cxxgen_extmods.py',
                                 '../src/normal/erfcx_funcs.cxx',
                                 '../src/normal/erfcx_funcs.h',
                                 '../src/normal/normal.h'],
                        output : ['erfcx_funcs_concrete.cxx',
                                  'erfcx_funcs_concrete.h',
                                  'erfcx_funcs.cxx',
                                  'erfcx_funcs.h',
                                  'normal_concrete.cxx',
                                  'normal_concrete.h',
                                  'normal.cxx',
                                  'normal.h'],
                        command : [py3, '@INPUT0@', '../src/normal', './ufunclab'])

py3.extension_module(
    'normal',
    mod_src,
    cpp_args : ['-std=c++14'],
    include_directories: [inc_np],
    dependencies: [np_dep, npymath_lib],
    install: true,
    subdir: 'ufunclab'
)

#----------------------------------------------------------------------
# Legacy NumPy .src templating -- gufuncs
#
# These extension modules are defined in a single .src file.
# These are templated files that are processed by the NumPy
# template language processor.  That processer is part of
# NumPy's deprecated distutils subpackage.  For ufunclab,
# the file `conv_template.py` (which implements the template
# processor) was copied from NumPy, tweaked a bit, and put
# in ufunclab/tools/conv_template.py.
#----------------------------------------------------------------------

numpy_templated_c_gufunc_dirs = [
    'backlash',
    'cross',
    'fillnan1d',
    'first',
    'hysteresis_relay',
    'linear_interp1d',
    'means',
    'peaktopeak',
    'searchsorted'
]

foreach src_dir : numpy_templated_c_gufunc_dirs

    module_name = '_' + src_dir                       # e.g. '_backlash'
    output = src_dir + '_gufunc.c'                    # e.g. 'backlash_gufunc.c'
    src_filename = output + '.src'                    # e.g. 'backlash_gufunc.c.src'
    src_pth = join_paths('../src', src_dir)           # e.g. '../src/backlash'
    src_fullpath = join_paths(src_pth, src_filename)  # e.g. '../src/blacklash/backlash_gufunc.c.src'

    mod_src = custom_target(
                  src_dir,
                  input : ['../tools/conv_template.py',
                           src_fullpath],
                  output : output,
                  command : [py3, '@INPUT0@', '@INPUT1@', './ufunclab'])

    py3.extension_module(
        module_name,
        [mod_src],
        include_directories: [inc_np],
        dependencies: [np_dep, npymath_lib],
        install: true,
        subdir: 'ufunclab'
    )

endforeach

#----------------------------------------------------------------------
# Legacy NumPy .src templating -- ufuncs
#
# The same as above, but for ufuncs.
# FIXME: DRY clean up needed here!
#----------------------------------------------------------------------

numpy_templated_c_ufunc_dirs = [
    'issnan',
]

foreach src_dir : numpy_templated_c_ufunc_dirs

    module_name = '_' + src_dir                       # e.g. '_issnan'
    output = src_dir + '_ufunc.c'                     # e.g. 'issnan_ufunc.c'
    src_filename = output + '.src'                    # e.g. 'issnan_ufunc.c.src'
    src_pth = join_paths('../src', src_dir)           # e.g. '../src/issnan'
    src_fullpath = join_paths(src_pth, src_filename)  # e.g. '../src/issnan/issnan_ufunc.c.src'

    mod_src = custom_target(
                  src_dir,
                  input : ['../tools/conv_template.py',
                           src_fullpath],
                  output : output,
                  command : [py3, '@INPUT0@', '@INPUT1@', './ufunclab'])

    py3.extension_module(
        module_name,
        [mod_src],
        include_directories: [inc_np],
        dependencies: [np_dep, npymath_lib],
        install: true,
        subdir: 'ufunclab'
    )

endforeach

#----------------------------------------------------------------------
# Pure C extension modules: no templating or code generation.
#----------------------------------------------------------------------

modules = [
    ['_cabssq',
        ['../src/cabssq/cabssq_ufunc.c']],
    ['_log1p',
        ['../src/log1p/log1p_ufunc.c']],
    ['_logfact',
        ['../src/logfactorial/logfactorial_ufunc.c',
         '../src/logfactorial/logfactorial.h',
         '../src/logfactorial/logfactorial.c']],
    ['_loggamma1p',
        ['../src/loggamma1p/loggamma1p_ufunc.c',
         '../src/loggamma1p/loggamma1p.h',
         '../src/loggamma1p/loggamma1p.c']],
    ['_gendot',
        ['../src/gendot/gendotmodule.c']],
    ['_ufunc_inspector',
        ['../src/ufunc-inspector/ufunc_inspector.c']]
]

foreach mod : modules

    py3.extension_module(
        mod[0],
        mod[1],
        include_directories: [inc_np],
        dependencies: [np_dep, npymath_lib],
        install: true,
        subdir: 'ufunclab'
    )

endforeach

#----------------------------------------------------------------------

py3.install_sources(
  ['__init__.py',
   '_gendot_wrap.py'],
  pure: false,
  subdir: 'ufunclab'
)

install_subdir('tests', install_dir: py3.get_install_dir() / 'ufunclab')
