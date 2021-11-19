# These are the names of the basic ufunkify opcodes.
basic_opcodes = [
    'RETURN',
    'PUSH_LOCAL',
    'STORE_LOCAL',
    'PUSH_CONSTANT',
    'PUSH_ZERO',
    'PUSH_FUNCTION',
    'CALL_FUNCTION',
    'UNARY_POSITIVE',
    'UNARY_NEGATIVE',
    'ADD',
    'SUBTRACT',
    'MULTIPLY',
    'TRUE_DIVIDE',
    'POWER',
    'SQUARE',
    'COMPARE_LT',
    'COMPARE_LE',
    'COMPARE_GT',
    'COMPARE_GE',
    'COMPARE_EQ',
    'COMPARE_NE',
    'JUMP',
    'JUMP_IF_FALSE',
    'JUMP_IF_TRUE',
]

# XXX Fix this duplication!
# Supported C99 functions (from <math.h>)

c99math = [
    ("fabs", 1),
    ("fmod", 2),
    ("fma", 3),
    ("fmax", 2),
    ("fmin", 2),
    ("fdim", 2),
    ("exp", 1),
    ("exp2", 1),
    ("expm1", 1),
    ("log", 1),
    ("log2", 1),
    ("log10", 1),
    ("log1p", 1),
    ("sqrt", 1),
    ("cbrt", 1),
    ("hypot", 2),
    ("pow", 2),
    ("sin", 1),
    ("cos", 1),
    ("tan", 1),
    ("asin", 1),
    ("acos", 1),
    ("atan", 1),
    ("atan2", 2),
    ("sinh", 1),
    ("cosh", 1),
    ("tanh", 1),
    ("asinh", 1),
    ("acosh", 1),
    ("atanh", 1),
    ("erf", 1),
    ("erfc", 1),
    ("lgamma", 1),
    ("tgamma", 1),
    ("ceil", 1),
    ("floor", 1),
    ("trunc", 1),
    ("round", 1),
    ("nearbyint", 1),
    ("rint", 1),
]

# Complex stuff is work in progress... not used yet.
complex_c99math = [
    # These commented-out entries return real values, not complex.
    #("cabs", 1),
    #("carg", 1),
    #("cimag", 1),
    #("creal", 1),
    #("cproj", 1),
    ("conj", 1),
    ("cexp", 1),
    ("clog", 1),
    ("csqrt", 1),
    ("cpow", 2),
    ("csin", 1),
    ("ccos", 1),
    ("ctan", 1),
    ("casin", 1),
    ("cacos", 1),
    ("catan", 1),
    ("csinh", 1),
    ("ccosh", 1),
    ("ctanh", 1),
    ("casinh", 1),
    ("cacosh", 1),
    ("catanh", 1),
]

#mathf_opcodes = [f'MATH_{name.upper()}F' for name, narg in c99math]
math_opcodes = [f'MATH_{name.upper()}' for name, narg in c99math]
#mathl_opcodes = [f'MATH_{name.upper()}L' for name, narg in c99math]

#opcodes = basic_opcodes + mathf_opcodes + math_opcodes + mathl_opcodes
opcodes = basic_opcodes + math_opcodes

def generate_opcode_files():
    """
    Generate two files, _opcode_names.py and _opcode_enum.h.
    """
    #
    # Write the Python file.
    #
    with open('_ufunkify_opcode_names.py', 'w') as f:
        f.write('''
# This file was generated automatically.  DO NOT EDIT!
# The names and values defined here must match those in the
# C enum defined in the file _opcode_enum.h.

# These are the ufunkify opcodes.
''')
        for i, name in enumerate(opcodes):
            f.write(f'{name:15} = {i:3}\n')
        f.write('\n')
        f.write(f'FIRST_MATH_OPCODE_INDEX = {len(basic_opcodes)}\n')
        f.write('\n')
        f.write('name_to_code = {\n')
        for name in opcodes:
            f.write(f'    "{name}" {" "*(15-len(name))}: {name},\n')
        f.write('}\n')
        f.write('\n')
        f.write('code_to_name = {\n')
        for name in opcodes:
            f.write(f'    {name} {" "*(15-len(name))}: "{name}",\n')
        f.write('}\n')

    #
    # Write the C header file.
    #
    with open('_ufunkify_opcodes.h', 'w') as f:
        f.write('''
#ifndef _OPCODES_H_
#define _OPCODES_H_

// This file was generated automatically.  DO NOT EDIT!
// The enum defined here must match names and values defined in the
// Python file _opcode_names.py.

// These are the ufunkify opcodes.
''')
        f.write('typedef enum {\n    ')
        f.write(',\n    '.join(opcodes))
        f.write('\n} opcode_t;\n\n')
        f.write(f'#define FIRST_MATH_OPCODE_INDEX {len(basic_opcodes)}\n\n')
        f.write('extern char *opcodes[];\n\n')
        f.write('#endif\n')

    with open('_ufunkify_opcodes.c', 'w') as f:
        f.write('''
// This file was generated automatically.  DO NOT EDIT!

// Table of opcode names as C strings.
''')
        f.write('char *opcodes[] = {\n    ')
        quoted_names = ['"' + name + '"' for name in opcodes]
        f.write(',\n    '.join(quoted_names))
        f.write('\n};\n')



def generate_math_lib_files():

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Write the Python file.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    with open('_ufunkify_math_lib_defs.py', 'w') as f:
        f.write('''
# This file was generated automatically.  DO NOT EDIT!
# The names and values defined here must match those in the
# corresponding C file.

''')
        f.write('c99math = {\n')
        for i, (name, nargs) in enumerate(c99math):
            f.write(f'    "{name}": {" "*(9 - len(name))} ({i:3}, {nargs:3}), \n')
        f.write('}\n')

        f.write('c99math_index_lookup = [\n')
        for i, (name, nargs) in enumerate(c99math):
            f.write(f'    ("{name}", {nargs}), \n')
        f.write(']\n\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Write the C files.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    with open('_ufunkify_c_function_types.h', 'w') as f:
        f.write('// This file was generated automatically.  DO NOT EDIT!\n\n')
        f.write('''
#ifndef _C_FUNCTION_TYPES_H_
#define _C_FUNCTION_TYPES_H_

#include <stdint.h>
#include <stdbool.h>
''')

        for typechar, csuffix, ctype in [('f', 'f', 'float'),
                                         ('d', '', 'double'),
                                         ('g', 'l', 'long double')]:
            # XXX Change int64_t to npy_intp.
            f.write(f'''
typedef {ctype} (*{ctype.replace(" ", "")}_function1_t)({ctype});
typedef {ctype} (*{ctype.replace(" ", "")}_function2_t)({ctype}, {ctype});
typedef {ctype} (*{ctype.replace(" ", "")}_function3_t)({ctype}, {ctype}, {ctype});
typedef void (*{ctype.replace(" ", "")}_loop_function_t)(char **args, int64_t *dimensions, int64_t *steps, void *data);

typedef struct _c_{ctype.replace(" ", "")}_function {{
    char name[16];
    int nargs;
    int math_lib_index;
    bool is_loop_function;
    union {{
        {ctype.replace(" ", "")}_function1_t function1;
        {ctype.replace(" ", "")}_function2_t function2;
        {ctype.replace(" ", "")}_function3_t function3;
        {ctype.replace(" ", "")}_loop_function_t loop_function;
    }} funcptr;
}} c_{ctype.replace(" ", "")}_function_t;

''')
        f.write('#endif\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    with open('_ufunkify_c_function_wrappers.c', 'w') as f:
        f.write('''
// This file was generated automatically.  DO NOT EDIT!

#include <stddef.h>
#include <math.h>
#include <stdbool.h>
#include "_ufunkify_c_function_types.h"

''')

        for typechar, csuffix, ctype in [('f', 'f', 'float'),
                                         ('d', '', 'double'),
                                         ('g', 'l', 'long double')]:
            f.write(f'c_{ctype.replace(" ", "")}_function_t c_{ctype.replace(" ", "")}_functions[] = {{\n')
            for k, (name, nargs) in enumerate(c99math):
                f.write(f'    {{.name = "{name}",{" "*(10-len(name))} .nargs = {nargs}, .math_lib_index = {k}, .is_loop_function = false, .funcptr.function{nargs} = &{name}{csuffix}}},\n')

            f.write('    {.name = "", .nargs = 0, .math_lib_index = -1, .is_loop_function = false, .funcptr.loop_function = NULL}\n};\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    with open('_ufunkify_c_function_wrappers.h', 'w') as f:
        f.write('''
// This file was generated automatically.  DO NOT EDIT!

#ifndef _C_FUNCTION_WRAPPERS_H_
#define _C_FUNCTION_WRAPPERS_H_

#include "_ufunkify_c_function_types.h"

''')

        for typechar, csuffix, ctype in [('f', 'f', 'float'),
                                         ('d', '', 'double'),
                                         ('g', 'l', 'long double')]:
            f.write(f'extern c_{ctype.replace(" ", "")}_function_t c_{ctype.replace(" ", "")}_functions[];\n')

        f.write("\n#endif\n")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    with open('_ufunkify_exec_math_switch_cases.h', 'w') as f:
        f.write('''
// This file was generated automatically.  DO NOT EDIT!
''')
        for name, nargs in c99math:
            if nargs == 1:
                f.write(
f'''
                case MATH_{name.upper()}:
                    datastack[(*datastack_index) - 1] = {name}(datastack[(*datastack_index) - 1]);
                    ++pc;
                    break;
''')
            elif nargs == 2:
                f.write(
f'''
                case MATH_{name.upper()}:
                    value2 = datastack[--(*datastack_index)];
                    datastack[(*datastack_index) - 1] = {name}(datastack[(*datastack_index) - 1], value2);
                    ++pc;
                    break;
''')
            elif nargs == 3:
                f.write(
f'''
                case MATH_{name.upper()}:
                    value3 = datastack[--(*datastack_index)];
                    value2 = datastack[--(*datastack_index)];
                    datastack[(*datastack_index) - 1] = {name}(datastack[(*datastack_index) - 1], value2, value3);
                    ++pc;
                    break;
''')

if __name__ == "__main__":
    generate_opcode_files()
    generate_math_lib_files()
