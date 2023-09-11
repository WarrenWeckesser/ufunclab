
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
# This necessary to avoid this build failure on Windows (using rtools 4.0 on
# the windows-2019 platform on github/actions)
#
#  [54/125] Linking target ufunclab/_next.cp310-win_amd64.pyd
#  FAILED: ufunclab/_next.cp310-win_amd64.pyd
#  "c++"  -o ufunclab/_next.cp310-win_amd64.pyd <...snip...>
#  c:/rtools40/ucrt64/bin/../lib/gcc/x86_64-w64-mingw32/10.3.0/../../../
#  ../x86_64-w64-mingw32/bin/ld.exe: c:/rtools40/ucrt64/bin/../lib/gcc/
#  x86_64-w64-mingw32/10.3.0/../../../../x86_64-w64-mingw32/lib/../lib/
#  libmsvcrt.a(deues00259.o):(.text+0x0): multiple definition of `nextafterf';
#  c:/rtools40/ucrt64/bin/../lib/gcc/x86_64-w64-mingw32/10.3.0/../../../../
#  x86_64-w64-mingw32/lib/../lib/libmingwex.a(lib64_libmingwex_a-nextafterf.o):
#  D:/a/rtools-testing/rtools-testing/mingw-w64-crt-git/src/mingw-w64/
#  mingw-w64-crt/math/nextafterf.c:10: first defined here
#
if cpp.get_id() != 'clang'
    add_project_link_arguments('-Wl,--allow-multiple-definition', language : 'cpp')
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
