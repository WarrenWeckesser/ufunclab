
#
# Enhancement ideas:
#
# Create unary versions of the binary operators that include
# the other constant operand in the ufunkified bytecode table.
# E.g. replace
#    PUSH_LOCAL     index = 0
#    PUSH_CONSTANT  constant = 5.0
#    MULTIPLY
# with
#    PUSH_LOCAL     index = 0
#    MULTIPLY_BY    constant = 5.0
# and replace
#    PUSH_ZERO
#    PUSH_LOCAL     index = 0
#    COMPARE_LT
# with
#    PUSH_LOCAL        index = 0
#    COMPARE_LT_VALUE  constant = 0.0
# To do this, it must be possible ensure that one of the
# operands on the stack will be a constant.  Perhaps code
# similar to the math function call rewrite or the stack size
# count could be used.

import warnings
import types
import dis
import copy
import numpy as np

# _ufunkify is the C extension module.
from ._ufunkify import _ufunkify

from . import _ufunkify_opcode_names as op

from ._ufunkify_math_lib_defs import c99math, c99math_index_lookup

c99math_d = dict(c99math)


_JUMP_OPCODES = [op.JUMP, op.JUMP_IF_FALSE, op.JUMP_IF_TRUE]

# The corresponding ufunkify op names are the same as these,
# but with 'BINARY_' removed.
py_binary_opnames = [
    'BINARY_ADD',
    'BINARY_SUBTRACT',
    'BINARY_MULTIPLY',
    'BINARY_TRUE_DIVIDE',
    'BINARY_POWER',
]


# This is the dtype of the numpy array that is passed to the
# _ufunkify function (the function in the C extension module).
# It is the "ufunkified" bytecode of the callable that is to be
# be converted to a ufunc.
program_dtype = np.dtype([('opcode', np.int32),
                          ('index', np.int32),
                          ('value', np.float64)])


def _create_program(f, namespaces=None, constants=None, namespace=None):
    """
    f must be a callable.

    namespaces must be a sequence.  Each element must be a module,
    or the string '<math.h>'.

    constants must be dictionary of name, value pairs for constants
    that will be used in the function. (XXX It should be possible for
    this to be done automatically...)

    """

    if namespaces is None:
        namespaces = []
    if constants is None:
        constants = {}
    program_table = []
    # num_func_pushes = 0
    # num_data_pushes = 0
    bc = dis.Bytecode(f)

    if namespace is not None:
        # This initial loop is just a test of an idea.  It doesn't
        # yet have any effect on what follows afterwards.
        found_names = []
        current_name = None
        for pc, instruction in enumerate(bc):
            if instruction.opname == 'LOAD_GLOBAL':
                if current_name is not None:
                    found_names.append(current_name)
                current_name = [instruction.argval]
            elif instruction.opname == 'LOAD_ATTR':
                assert current_name is not None
                current_name.append(instruction.argval)
            elif instruction.opname in ['LOAD_METHOD', 'LOAD_FUNCTION']:
                current_name.append(instruction.argval)
                found_names.append(current_name)
                current_name = None
            else:
                if current_name is not None:
                    found_names.append(current_name)
                    current_name = None
        print("Found these names:")
        print(found_names)
        found_objects = []
        for found_name in found_names:
            top = found_name[0]
            if top not in namespace:
                if top not in c99math:
                    raise NameError('name %r is not defined' % top)
                if len(found_name) > 1:
                    raise AttributeError(f'C99 math function {top} has no '
                                         f'attribute {found_name[2]}')
                obj = ('<math.h>', top)
            else:
                obj = namespace[top]
                attrs = found_name[1:][::-1]
                while attrs:
                    attr = attrs.pop()
                    obj = getattr(obj, attr)
            found_objects.append(obj)
        print("Found objects:", found_objects)

    for pc, instruction in enumerate(bc):
        # instruction is a *Python* bytecode instruction (as represented by
        # dis.Bytecode(f)).
        if instruction.opname in py_binary_opnames:
            program_table.append((op.name_to_code[instruction.opname[7:]],
                                  0, 0.0))
        elif instruction.opname in ['UNARY_POSITIVE', 'UNARY_NEGATIVE']:
            program_table.append((op.name_to_code[instruction.opname],
                                  0, 0.0))
        elif instruction.opname == 'LOAD_GLOBAL':
            if instruction.argval in constants:
                program_table.append((op.PUSH_CONSTANT, 0,
                                      constants[instruction.argval]))
            else:
                # Anything else global is expected to be a function.
                funcname = instruction.argval
                found = False
                # Find the function.
                for namespace in namespaces:
                    # Special case check for <math.h>
                    if namespace == '<math.h>':
                        # Rename to corresponding name in C99 math.h
                        funcname2 = {'minimum': 'fmin',
                                     'maximum': 'fmax',
                                     'min': 'fmin',
                                     'max': 'fmax',
                                     'arctan2': 'atan2'}.get(funcname,
                                                             funcname)
                        if funcname2 in c99math_d:
                            mathfuncindex, nargs = c99math_d[funcname2]
                            if nargs == 1:
                                program_table.append((op.PUSH_FUNCTION,
                                                      mathfuncindex,
                                                      0.0))
                                found = True
                                break
                            elif nargs == 2:
                                program_table.append((op.PUSH_FUNCTION,
                                                      mathfuncindex,
                                                      0.0))
                                found = True
                                break
                            elif nargs == 3:
                                program_table.append((op.PUSH_FUNCTION,
                                                      mathfuncindex,
                                                      0.0))
                                found = True
                                break
                            else:
                                raise RuntimeError('functions with more than '
                                                   '3 args not handled yet.')
                        else:
                            continue
                    else:
                        assert isinstance(namespace, types.ModuleType)
                        func = getattr(namespace, funcname)
                        if func is None:
                            continue
                        if not isinstance(func, np.ufunc):
                            raise RuntimeError('%r is not a ufunc.' % funcname)
                        if func.nout != 1:
                            raise RuntimeError("nout != 1 for "
                                               f"{instruction.opname}")
                        sigstr_d = 'd'*func.nin + '->d'
                        if sigstr_d not in func.types:
                            raise RuntimeError(f"{funcname} does not have a "
                                               f"loop for {sigstr_d}")
                        # This won't work--currently only handle <math.h>.
                        program_table.append((op.PUSH_FUNCTION, func, funcname,
                                              namespace.__name__))
                        found = True
                if not found:
                    raise NameError('name %r is not defined.' % funcname)
        elif instruction.opname == 'LOAD_CONST':
            program_table.append((op.PUSH_CONSTANT, 0,
                                  float(instruction.argval)))
        elif instruction.opname == 'LOAD_FAST':
            program_table.append((op.PUSH_LOCAL, instruction.arg, 0.0))
        elif instruction.opname == 'STORE_FAST':
            program_table.append((op.STORE_LOCAL, instruction.arg, 0.0))
        elif instruction.opname == 'CALL_FUNCTION':
            program_table.append((op.CALL_FUNCTION, instruction.argval, 0.0))
        elif instruction.opname == 'COMPARE_OP':
            if instruction.argval == '<':
                program_table.append((op.COMPARE_LT, 0, 0.0))
            elif instruction.argval == '<=':
                program_table.append((op.COMPARE_LE, 0, 0.0))
            elif instruction.argval == '>':
                program_table.append((op.COMPARE_GT, 0, 0.0))
            elif instruction.argval == '>=':
                program_table.append((op.COMPARE_GE, 0, 0.0))
            elif instruction.argval == '==':
                program_table.append((op.COMPARE_EQ, 0, 0.0))
            elif instruction.argval == '!=':
                program_table.append((op.COMPARE_NE, 0, 0.0))
            else:
                raise RuntimeError('unknown comparison operator '
                                   f'{instruction.argval}')
        elif instruction.opname == 'JUMP_FORWARD':
            # Python's JUMP_FORWARD is a relative jump.  Convert that
            # to a ufunkify absolute jump.
            offset = instruction.arg // 2 + 1
            program_table.append((op.JUMP, pc + offset, 0.0))
        elif instruction.opname == 'POP_JUMP_IF_FALSE':
            dest = instruction.argval // 2
            program_table.append((op.JUMP_IF_FALSE, dest, 0.0))
        elif instruction.opname == 'POP_JUMP_IF_TRUE':
            dest = instruction.argval // 2
            program_table.append((op.JUMP_IF_TRUE, dest, 0.0))
        elif instruction.opname == 'RETURN_VALUE':
            program_table.append((op.RETURN, 0, 0.0))
        elif instruction.opname == 'STORE_FAST':
            program_table.append((op.STORE_LOCAL, 0, 0.0))
        else:
            raise RuntimeError('unhandled op %r' % instruction.opname)

    program_table = np.array(program_table, dtype=program_dtype)
    return program_table


def map_calls_to_push_functions(table, index=0, callmap=None, funcstack=None):
    if callmap is None:
        callmap = {}
    if funcstack is None:
        funcstack = []

    if index == len(table):
        warnings.warn("Fall through return in callable. "
                      "This should not happen?")
        return callmap, funcstack

    opcode, operand_index, operand_value = table[index]

    # print(index, op.code_to_name[opcode], operand_index)

    if opcode == op.RETURN:
        return callmap, funcstack

    if opcode == op.PUSH_FUNCTION:
        funcstack.append(index)
        callmap, funcstack = map_calls_to_push_functions(table, index + 1,
                                                         callmap, funcstack)
        return callmap, funcstack

    if opcode == op.CALL_FUNCTION:
        if len(funcstack) == 0:
            raise RuntimeError('CALL_FUNCTION with an empty function stack')
        push_function_index = funcstack.pop()
        callmap.setdefault(index, []).append(push_function_index)
        callmap, funcstack = map_calls_to_push_functions(table, index + 1,
                                                         callmap, funcstack)
        return callmap, funcstack

    if opcode == op.JUMP:
        return map_calls_to_push_functions(table, operand_index, callmap,
                                           funcstack)

    if opcode in [op.JUMP_IF_TRUE, op.JUMP_IF_FALSE]:
        callmap1, funcstack1 = \
            map_calls_to_push_functions(table, index + 1,
                                        copy.deepcopy(callmap),
                                        funcstack.copy())
        callmap2, funcstack2 = \
            map_calls_to_push_functions(table, operand_index,
                                        copy.deepcopy(callmap),
                                        funcstack.copy())
        assert funcstack1 == funcstack2
        for key, value in callmap2.items():
            callmap1.setdefault(key, []).extend(value)
        return callmap1, funcstack1

    # All other opcodes.
    callmap, funcstack = map_calls_to_push_functions(table, index + 1,
                                                     callmap, funcstack)
    return callmap, funcstack


def compute_stack_size(table, index=0, datastack_size=0, funcstack_size=0):
    """
    Returns upper bounds on the depths of the data stack and the function stack
    for the program in `table`.
    """
    # This recursive algorithm should work because we don't allow loops.

    if index == len(table):
        # This should not happen?  It means a "fall through" return.
        print("WARNING! Fall through return!")
        return datastack_size, funcstack_size

    opcode, operand_index, operand_value = table[index]

    # print(op.code_to_name[opcode], operand_index)

    if opcode == op.RETURN:
        return datastack_size, funcstack_size

    if opcode in [op.UNARY_POSITIVE, op.UNARY_NEGATIVE]:
        # No net change to the stacks.
        ds, fs = compute_stack_size(table, index + 1, datastack_size,
                                    funcstack_size)
        return max(ds, datastack_size), max(fs, funcstack_size)

    if opcode in [op.PUSH_LOCAL, op.PUSH_CONSTANT, op.PUSH_ZERO]:
        # datastack increases by 1.
        ds, fs = compute_stack_size(table, index + 1, datastack_size + 1,
                                    funcstack_size)
        return max(ds, datastack_size), max(fs, funcstack_size)

    if opcode in [op.STORE_LOCAL, op.ADD, op.SUBTRACT, op.MULTIPLY,
                  op.TRUE_DIVIDE, op.POWER, op.COMPARE_LT, op.COMPARE_LE,
                  op.COMPARE_GT, op.COMPARE_GE, op.COMPARE_EQ, op.COMPARE_NE]:
        # datastack decreases by 1.
        ds, fs = compute_stack_size(table, index + 1, datastack_size - 1,
                                    funcstack_size)
        return max(ds, datastack_size), max(fs, funcstack_size)

    if opcode == op.PUSH_FUNCTION:
        # funcstack increases by 1.
        ds, fs = compute_stack_size(table, index + 1, datastack_size,
                                    funcstack_size + 1)
        return max(ds, datastack_size), max(fs, funcstack_size)

    if opcode == op.CALL_FUNCTION:
        # datastack decreases by nargs-1, funcstack decreases by 1.
        ds, fs = compute_stack_size(table, index + 1,
                                    datastack_size - operand_index + 1,
                                    funcstack_size - 1)
        return max(ds, datastack_size), max(fs, funcstack_size)

    if opcode in [op.JUMP_IF_TRUE, op.JUMP_IF_FALSE]:
        branch1_datastack_size, branch1_funcstack_size = \
            compute_stack_size(table, index + 1, datastack_size - 1,
                               funcstack_size)
        branch2_datastack_size, branch2_funcstack_size = \
            compute_stack_size(table, operand_index, datastack_size - 1,
                               funcstack_size)
        ds = max(branch1_datastack_size, branch2_datastack_size)
        fs = max(branch1_funcstack_size, branch2_funcstack_size)
        return max(ds, datastack_size), max(fs, funcstack_size)

    if opcode == op.JUMP:
        ds, fs = compute_stack_size(table, operand_index, datastack_size,
                                    funcstack_size)
        return max(ds, datastack_size), max(fs, funcstack_size)

    # Must be a MATH operation.
    math_lib_index = opcode - op.FIRST_MATH_OPCODE_INDEX
    nargs = c99math_index_lookup[math_lib_index][1]
    # datastack decreases by nargs-1, funcstack does not change.
    ds, fs = compute_stack_size(table, index + 1, datastack_size - nargs + 1,
                                funcstack_size)
    return max(ds, datastack_size), max(fs, funcstack_size)


def table_to_text(table):
    text = []
    for row in table:
        text.append((op.code_to_name[row['opcode']],
                     row['index'], row['value']))
    return text


def remove_index(table, index):
    for pc, (opcode, operand_index, operand_value) in enumerate(table):
        if opcode in _JUMP_OPCODES:
            if operand_index > index:
                table[pc]['index'] = operand_index - 1
    table = np.concatenate((table[:index], table[index+1:]))
    return table


def remove_noops(table):
    NOOP = op.UNARY_POSITIVE
    noops = table['opcode'] == NOOP
    # First adjust any jumps so their targets are not no-ops.
    for pc, (opcode, operand_index, operand_value) in enumerate(table):
        if opcode in _JUMP_OPCODES:
            while operand_index < len(table) and noops[operand_index]:
                operand_index += 1
            table[pc]['index'] = operand_index
    #
    for pc, (opcode, operand_index, operand_value) in enumerate(table):
        if opcode in _JUMP_OPCODES:
            # Count no-ops between pc and the destination.
            if operand_index > pc:
                num_noops = noops[:operand_index].sum()
                table[pc]['index'] -= num_noops
            else:
                raise RuntimeError('unexpected non-forward jump')

    table = table[~noops]
    return table


def rewrite_table(table):
    # New idea:
    # Identify sequential blocks of code.  These are blocks with no JUMPs,
    # and no destinations of JUMPs (i.e. there are no JUMPs *into* the block).
    # Within each block, PUSH/CALL pairs can be nested or not, just like
    # parentheses in an expression.  This potential nesting must be handled
    # when finding the matching pairs of PUSH/CALL operations.  E.g.
    # exp(fabs(x)) is
    #  +----PUSH_FUNCTION     function = exp,  nargs = 1, math_lib_index = 6
    #  |  +-PUSH_FUNCTION     function = fabs,  nargs = 1, math_lib_index = 0
    #  |  | PUSH_LOCAL        index = 0
    #  |  +-CALL_FUNCTION
    #  +----CALL_FUNCTION
    #       RETURN
    # and arctan2(sin(x), cos(2*x)) is
    #  +----PUSH_FUNCTION     function = atan2,  nargs = 2, math_lib_index = 23
    #  |  +-PUSH_FUNCTION     function = sin,  nargs = 1, math_lib_index = 17
    #  |  | PUSH_LOCAL        index = 0
    #  |  +-CALL_FUNCTION
    #  |  +-PUSH_FUNCTION     function = cos,  nargs = 1, math_lib_index = 18
    #  |  | PUSH_CONSTANT     constant =       2.000000000000
    #  |  | PUSH_LOCAL        index = 0
    #  |  | MULTIPLY
    #  |  +-CALL_FUNCTION
    #  +----CALL_FUNCTION
    #       RETURN
    #

    # Why does using this rewrite rule make the code *slower*?
    # That's what happened with
    #     h = lambda x: 0 if x < 0 else 1
    # applied to x = np.random.randn(1000000).
    for pc, operation in enumerate(table):
        if operation[0] == op.PUSH_CONSTANT and operation[2] == 0:
            table[pc]['opcode'] = op.PUSH_ZERO

    callmap, funcstack = map_calls_to_push_functions(table)

    fixlist = [(call_index, push_index_list[0])
               for call_index, push_index_list in callmap.items()
               if len(set(push_index_list)) == 1]
    while len(fixlist) > 0:
        call_index, push_index = fixlist.pop()
        func_opcode, func_operand_index, func_operand_value = table[push_index]
        math_opcode = func_operand_index + op.FIRST_MATH_OPCODE_INDEX
        table[call_index]['opcode'] = math_opcode
        # XXX Not done--need to remove the PUSH_FUNCTION opcode at push_index.
        table[push_index]['opcode'] = op.UNARY_POSITIVE  # Serves as a no-op.

    # Replace
    #     LOAD_CONSTANT  2
    #     POWER
    # with
    #     SQUARE
    is_target = np.zeros(len(table), dtype=np.bool8)
    for pc, operation in enumerate(table):
        if operation[0] in _JUMP_OPCODES:
            is_target[operation[1]] = True
    for pc, operation in enumerate(table):
        if (pc > 0
                and operation[0] == op.POWER
                and not is_target[pc]
                and table[pc-1][0] == op.PUSH_CONSTANT
                and table[pc-1][2] == 2):
            table[pc-1]['opcode'] = op.UNARY_POSITIVE  # No-op
            table[pc]['opcode'] = op.SQUARE

    table = remove_noops(table)

    return table


def ufunkify(func, namespaces=('<math.h>',), constants=None, rewrite=False,
             name=None, namespace=None):
    """
    Convert the lambda object `func` into a NumPy ufunc.

    Parameters
    ----------
    func : callable
        Callable to converted to a ufunc.
    namespaces :
    constants :
    rewrite : bool, optional
        If True, apply some optimizations to the ufunkified bytecode.
    name :
    namespace :
        Experimental (but then again, so is everything else).
    """
    if not callable(func):
        raise ValueError('func must be callable.')

    if name is not None:
        ufunk_name = name
    else:
        ufunk_name = getattr(func, '__name__', f'<callable-{id(func)}>')

    if namespaces is not None:
        if list(namespaces) != ['<math.h>']:
            raise ValueError('The only namespace implemented is "<math.h>".')

    nargs = func.__code__.co_argcount
    local_size = func.__code__.co_nlocals

    program_table = _create_program(func, namespaces, constants,
                                    namespace=namespace)

    if rewrite:
        program_table = rewrite_table(program_table)

    datastack_size, funcstack_size = compute_stack_size(program_table)

    ufunk = _ufunkify(ufunk_name, nargs, local_size, datastack_size,
                      funcstack_size, program_table)

    return ufunk
