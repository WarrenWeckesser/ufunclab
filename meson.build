
project(
  'ufunclab',
  'c', 'cpp',
  meson_version: '>= 1.1.0',
  default_options: [
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

# Hack attempt to fix Windows build on CI.
add_global_arguments('-Wl,allow-multiple-definitiions', language : 'cpp')

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()

# Generate _version.py by reading the version from pyproject.toml.
custom_target(input : 'pyproject.toml',
              output : '_version.py',
              command : [
                py3, '../_build_utils/make_version_file.py',
                '@INPUT@', '@OUTPUT@' 
              ],
              install : true,
              install_dir : py3.get_install_dir() / 'ufunclab')

subdir('ufunclab')
