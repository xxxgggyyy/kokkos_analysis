import os
import ycm_core
 
flags = [
    '-Wall',
    '-Wextra',
    '-Werror',
    '-Wno-long-long',
    '-Wno-variadic-macros',
    '-fexceptions',
    '-DNDEBUG',
    '-std=c++14',
    '-x',
    'c++',
    '-I',
    '/usr/include',
    '-I',
    '/home/xui/misc/01_hello_world',
    '-I',
    '/home/xui/misc/kokkos_analysis/core/src',
    '-I',
    '/home/xui/misc/kokkos_analysis/tpls/desul/include'
  ]
 
SOURCE_EXTENSIONS = [ '.cpp', '.cxx', '.cc', '.c', '.cu']
 
def Settings(**kwargs ):
  return {
    'flags': flags,
    'do_cache': True
  }
