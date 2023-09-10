
project(
  'ufunclab',
  'c', 'cpp',
  meson_version: '>= 1.1.0',
  default_options: [
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Hack attempt to fix Windows build on CI.
if cpp.get_id() != 'clang'
    add_project_link_arguments('-Wl,--allow-multiple-definition', language : 'cpp')
endif

# We need -lm for all C code.
c_m_dep = cc.find_library('m', required : false)
if c_m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# C++ shouldn't need -lm, but the Windows build seems to be missing
# 'feraiseexcept', so let's try...
cpp_m_dep = cpp.find_library('m', required : false)
if cpp_m_dep.found()
  add_project_link_arguments('-lm', language : 'cpp')
endif

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
