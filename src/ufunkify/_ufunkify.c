#include <stdio.h>
#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "_ufunkify_c_function_types.h"
#include "_ufunkify_c_function_wrappers.h"
#include "_ufunkify_opcodes.h"


//
// This is the structure of the data in the NumPy array passed
// to the Python function ufunkify() from the Python wrapper.
// It matches the array's dtype:
//     np.dtype([('opcode', np.int32),
//               ('index', np.int32),
//               ('value', np.float64)])
//
typedef struct _c99_double_program_table {
    int32_t opcode;
    int32_t index;
    double value;
} simple_operation_double_t;


//
// This is the structure of the "statements" in the mini
// stack-based language used to implement the chained
// ufuncs.  opcode_t is an enum defined in _opcodes.h.
//
typedef struct _double_operation {
    opcode_t opcode;
    union {
         // PUSH_LOCAL, STORE_LOCAL
        int32_t index;

        // PUSH_CONSTANT
        double constant;

        // JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE
        int32_t jump_destination;

        // C function data
        c_double_function_t function;
    } operand;
} operation_double_t;


void print_program(operation_double_t *program, int program_len)
{
    for (int i = 0; i < program_len; ++i) {
        printf("%4d: %-16s ", i, opcodes[program[i].opcode]);
        switch (program[i].opcode) {
            case PUSH_LOCAL:
            case STORE_LOCAL:
                printf(" index = %d", program[i].operand.index);
                break;

            case PUSH_CONSTANT:
                printf(" constant = %20.12f", program[i].operand.constant);
                break;

            case JUMP:
            case JUMP_IF_FALSE:
            case JUMP_IF_TRUE:
                printf(" destination = %d", program[i].operand.jump_destination);
                break;

            case PUSH_FUNCTION:
                // The only case handled at the moment: dp->index is the index
                // into the array of C double function data.
                printf(" function = %s,  nargs = %d, math_lib_index = %d",
                       program[i].operand.function.name,
                       program[i].operand.function.nargs,
                       program[i].operand.function.math_lib_index);
                break;

            case CALL_FUNCTION:
            case UNARY_POSITIVE:
            case UNARY_NEGATIVE:
            case ADD:
            case SUBTRACT:
            case MULTIPLY:
            case TRUE_DIVIDE:
            case SQUARE:
            case POWER:
            case COMPARE_LT:
            case COMPARE_LE:
            case COMPARE_GT:
            case COMPARE_GE:
            case COMPARE_EQ:
            case COMPARE_NE:
            case RETURN:
            default:
                break;
        }
        printf("\n");
    }
}


static inline int call_double_func(c_double_function_t *fp,
                                   double *stack, int stack_size, int *stack_index)
{
    //printf("call_double_func: name = %s   nargs = %d\n", fp->name, fp->nargs);

    switch (fp->nargs) {
        case 1:
            if (fp->is_loop_function) {
                // Call a loop function with a single set of inputs to compute
                // a single output.
            }
            else {
                // Directly call the function.
                //printf("call_double_func: *stack_index = %d\n", *stack_index);
                stack[(*stack_index) - 1] = (fp->funcptr.function1)(stack[(*stack_index) - 1]);
                //printf("result = %f\n", stack[(*stack_index) - 1]);

                return 0;
            }
            break;
        case 2:
            if (fp->is_loop_function) {
                //
            }
            else {
                // Directly call the function.
                double value2 = stack[--(*stack_index)];
                stack[(*stack_index) - 1] = (fp->funcptr.function2)(stack[(*stack_index) - 1], value2);
                return 0;
            }
            break;
        case 3:
            if (fp->is_loop_function) {
                //
            }
            else {
                double value3 = stack[--(*stack_index)];
                double value2 = stack[--(*stack_index)];
                stack[(*stack_index) - 1] = (fp->funcptr.function3)(stack[(*stack_index) - 1], value2, value3);
                return 0;
            }
            break;
        default:
            // This must not happen!
            return -1;
    }
    return -2;
}

//
// Execute a program.
//
int exec_double(operation_double_t *program, int program_size,
                double *local, int local_size,
                double *datastack, int datastack_size, int *datastack_index,
                c_double_function_t *funcstack, int funcstack_size, int *funcstack_index)
{
    int pc = 0;
    int status;

    while (true) {
        c_double_function_t func;
        double value2, value3;

        //printf("%6d %-15s:\n", pc, opcode_names[program[pc].opcode]);

        switch (program[pc].opcode) {
            case RETURN:
                //printf("%6d RETURN:\n", pc);
                return 0;
                break;  // just a formality
            case PUSH_CONSTANT:
                //printf("%6d PUSH_CONSTANT:\n", pc);
                datastack[(*datastack_index)++] = program[pc].operand.constant;
                //printf("After PUSH_CONSTANT: top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case PUSH_ZERO:
                //printf("%6d PUSH_ZERO:\n", pc);
                datastack[(*datastack_index)++] = 0;
                //printf("After PUSH_CONSTANT: top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case PUSH_LOCAL:
                //printf("%6d PUSH_LOCAL:\n", pc);
                datastack[(*datastack_index)++] = local[program[pc].operand.index];
                //printf("After PUSH_LOCAL: top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case STORE_LOCAL:
                //printf("%6d STORE_LOCAL:\n", pc);
                local[program[pc].operand.index] = datastack[--(*datastack_index)];
                ++pc;
                break;
            case UNARY_POSITIVE:
                //printf("%6d UNARY_POSITIVE:\n", pc);
                // It's a no-op.
                ++pc;
                break;
            case UNARY_NEGATIVE:
                //printf("%6d UNARY_NEGATIVE:\n", pc);
                datastack[(*datastack_index) - 1] = -datastack[(*datastack_index) - 1];
                ++pc;
                break;
            case ADD:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] += value2;
                //printf("After ADD:           top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case SUBTRACT:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] -= value2;
                //printf("After SUBTRACT:      top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case MULTIPLY:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] *= value2;
                ++pc;
                break;
            case TRUE_DIVIDE:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] /= value2;
                ++pc;
                break;
            case POWER:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = pow(datastack[(*datastack_index) - 1], value2);
                //printf("After POWER:         top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case SQUARE:
                datastack[(*datastack_index) - 1] *= datastack[(*datastack_index) - 1];
                ++pc;
                break;
            case PUSH_FUNCTION:
                funcstack[(*funcstack_index)++] = program[pc].operand.function;
                ++pc;
                break;
            case CALL_FUNCTION:
                func = funcstack[--(*funcstack_index)];
                //printf("CALL_FUNCTION: before call: *datastack_index = %d\n", *datastack_index);
                //printf("CALL_FUNCTION: before call: top of stack = %f\n", datastack[(*datastack_index) - 1]);
                status = call_double_func(&func, datastack, datastack_size, datastack_index);
                //printf("After CALL_FUNCTION: top of datastack: %f\n", datastack[(*datastack_index) - 1]);
                ++pc;
                break;
            case COMPARE_LT:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = datastack[(*datastack_index) - 1] < value2;
                ++pc;
                break;
            case COMPARE_LE:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = datastack[(*datastack_index) - 1] <= value2;
                ++pc;
                break;
            case COMPARE_GT:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = datastack[(*datastack_index) - 1] > value2;
                ++pc;
                break;
            case COMPARE_GE:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = datastack[(*datastack_index) - 1] >= value2;
                ++pc;
                break;
            case COMPARE_EQ:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = datastack[(*datastack_index) - 1] == value2;
                ++pc;
                break;
            case COMPARE_NE:
                value2 = datastack[--(*datastack_index)];
                datastack[(*datastack_index) - 1] = datastack[(*datastack_index) - 1] != value2;
                ++pc;
                break;
            case JUMP:
                //printf("JUMP: desination = %d\n", program[pc].operand.jump_desination);
                pc = program[pc].operand.jump_destination;
                break;
            case JUMP_IF_FALSE:
                if (!datastack[--(*datastack_index)]) {
                    pc = program[pc].operand.jump_destination;
                }
                else {
                    ++pc;
                }
                break;
            case JUMP_IF_TRUE:
                if (datastack[--(*datastack_index)]) {
                    pc = program[pc].operand.jump_destination;
                }
                else {
                    ++pc;
                }
                break;

#include "_ufunkify_exec_math_switch_cases.h"

            default:
                // Unknown opcode.  This should not happen!
                printf("ERROR: unknown opcode %d\n at pc=%d\n", program[pc].opcode, pc);
                return -2;
        }
        if (pc == program_size) {
            // end of program with no RETURN
            return -1;
        }
    }
}


static void loop_d_d(char **args, npy_intp *dimensions,
                     npy_intp* steps, void* data)
{
    //printf("loop_d_d: starting\n");

    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out = args[1];
    npy_intp in_step1 = steps[0];
    npy_intp out_step = steps[1];

    //printf("loop_d_d: data = %lld\n", data);

    // Unpack data into local array sizes and the array of operation_double_t.
    int32_t* sizes = (int32_t *) data;
    int32_t local_size     = sizes[0];
    int32_t datastack_size = sizes[1];
    int32_t funcstack_size = sizes[2];
    int32_t program_size   = sizes[3];
    operation_double_t* program = (operation_double_t *) &sizes[4];

    // Note: these are variable length arrays (VLAs).  VLAs are supported in
    // C99, but in not necessarily in C11.  Technically, the code should check
    // that __STDC_NO_VLA__ is not 1 before making these declarations.  If that
    // macro is 1, malloc/free should be used instead.
    double local[local_size];
    double datastack[datastack_size];
    c_double_function_t funcstack[funcstack_size];

    for (i = 0; i < n; i++) {
        local[0] = *(double *) in1;

        int datastack_index = 0;
        int funcstack_index = 0;

        int status = exec_double(program, program_size,
                                 local, local_size,
                                 datastack, datastack_size, &datastack_index,
                                 funcstack, funcstack_size, &funcstack_index);
        // Should check that:
        //     status == 0
        //     funcstack_index == 0
        //     datastack_index == 1
        // but what to do if the checks fail?
        if (status != 0) {
            fprintf(stderr, "loop_d_d: exec_double returned %d\n", status);
        }
        *((double *)out) = datastack[0];

        in1 += in_step1;
        out += out_step;
    }

    //printf("loop_d_d: returning\n");

}


static void loop_dd_d(char **args, npy_intp *dimensions,
                      npy_intp* steps, void* data)
{
    //printf("loop_dd_d: starting\n");

    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *out = args[2];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp out_step = steps[2];

    //printf("loop_dd_d: data = %lld\n", data);

    // Unpack data into local array sizes and the array of operation_double_t.
    int32_t* sizes = (int32_t *) data;
    int32_t local_size     = sizes[0];
    int32_t datastack_size = sizes[1];
    int32_t funcstack_size = sizes[2];
    int32_t program_size   = sizes[3];
    operation_double_t* program = (operation_double_t *) &sizes[4];

    // Note: these are variable length arrays (VLAs).  VLAs are supported in
    // C99, but in not necessarily in C11.  Technically, the code should check
    // that __STDC_NO_VLA__ is not 1 before making these declarations.  If that
    // macro is 1, malloc/free should be used instead.
    double local[local_size];
    double datastack[datastack_size];
    c_double_function_t funcstack[funcstack_size];

    for (i = 0; i < n; i++) {
        local[0] = *(double *) in1;
        local[1] = *(double *) in2;

        int datastack_index = 0;
        int funcstack_index = 0;

        int status = exec_double(program, program_size,
                                 local, local_size,
                                 datastack, datastack_size, &datastack_index,
                                 funcstack, funcstack_size, &funcstack_index);
        // Should check that:
        //     status == 0
        //     funcstack_index == 0
        //     datastack_index == 1
        // but what to do if the checks fail?
        if (status != 0) {
            fprintf(stderr, "loop_dd_d: exec_double returned %d\n", status);
        }
        *((double *)out) = datastack[0];

        in1 += in_step1;
        in2 += in_step2;
        out += out_step;
    }

    //printf("loop_dd_d: returning\n");

}


static void loop_ddd_d(char **args, npy_intp *dimensions,
                      npy_intp* steps, void* data)
{
    //printf("loop_dd_d: starting\n");

    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *in3 = args[2];
    char *out = args[3];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp in_step3 = steps[2];
    npy_intp out_step = steps[3];

    //printf("loop_ddd_d: data = %lld\n", data);

    // Unpack data into local array sizes and the array of operation_double_t.
    int32_t* sizes = (int32_t *) data;
    int32_t local_size     = sizes[0];
    int32_t datastack_size = sizes[1];
    int32_t funcstack_size = sizes[2];
    int32_t program_size   = sizes[3];
    operation_double_t* program = (operation_double_t *) &sizes[4];

    // Note: these are variable length arrays (VLAs).  VLAs are supported in
    // C99, but in not necessarily in C11.  Technically, the code should check
    // that __STDC_NO_VLA__ is not 1 before making these declarations.  If that
    // macro is 1, malloc/free should be used instead.
    double local[local_size];
    double datastack[datastack_size];
    c_double_function_t funcstack[funcstack_size];

    for (i = 0; i < n; i++) {
        local[0] = *(double *) in1;
        local[1] = *(double *) in2;
        local[2] = *(double *) in3;

        int datastack_index = 0;
        int funcstack_index = 0;

        int status = exec_double(program, program_size,
                                 local, local_size,
                                 datastack, datastack_size, &datastack_index,
                                 funcstack, funcstack_size, &funcstack_index);
        // Should check that:
        //     status == 0
        //     funcstack_index == 0
        //     datastack_index == 1
        // but what to do if the checks fail?
        if (status != 0) {
            fprintf(stderr, "loop_ddd_d: exec_double returned %d\n", status);
        }
        *((double *)out) = datastack[0];

        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        out += out_step;
    }

    //printf("loop_ddd_d: returning\n");

}

// This will be made a copy of PyUFunc_Type, with the tp_dealloc
// and tp_repr slots changed.
//PyTypeObject PyUFunk_Type;


typedef struct _derived_ufunc_type {
    PyTypeObject ufunc_type;
    destructor parent_tp_dealloc;
    reprfunc   parent_tp_repr;
} DerivedUFuncTypeObject;

DerivedUFuncTypeObject PyUFunk_Type;


static PyObject *
_ufunkify(PyObject *self, PyObject *args)
{
    int nargs;
    char *name;
    int local_size, datastack_size, funcstack_size;
    PyArrayObject *prog_array;
    size_t program_len;
    simple_operation_double_t *data;
    operation_double_t *program;
    PyObject *ufunc;

    // ufunc_functions and ufunc_data are the arrays that will be
    // passed to PyUFunc_FromFuncAndData.
    PyUFuncGenericFunction *ufunc_functions = malloc(sizeof(PyUFuncGenericFunction *));
    void **ufunc_data = malloc(sizeof(void *));
    if (!ufunc_data || !ufunc_functions) {
        PyErr_SetString(PyExc_MemoryError, "_ufunkify: failed to allocate memory.");
        free(ufunc_functions);
        free(ufunc_data);
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "siiiiO", &name, &nargs, &local_size, &datastack_size,
                                         &funcstack_size, &prog_array)) {
        free(ufunc_functions);
        free(ufunc_data);
        return NULL;
    }

    if (!PyArray_Check(prog_array)) {
        PyErr_SetString(PyExc_ValueError, "fourth argument must be a numpy array.");
        free(ufunc_functions);
        free(ufunc_data);
        return NULL;
    }
    if (nargs != 1 && nargs != 2 && nargs != 3) {
        PyErr_SetString(PyExc_ValueError,
                        "only callables with 1, 2 or 3 arguments are handled.");
        free(ufunc_functions);
        free(ufunc_data);
        return NULL;
    }

    program_len = PyArray_DIM(prog_array, 0);
    void * mem = malloc(sizeof(int32_t)*4 + sizeof(operation_double_t)*program_len);

    if (mem == NULL) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate memory.");
        free(ufunc_functions);
        free(ufunc_data);
        return NULL;
    }

    // XXX All the usual warnings about strcpy...
    char *copyname = malloc(strlen(name)+ 1);
    if (!copyname) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate memory.");
        free(mem);
        free(ufunc_functions);
        free(ufunc_data);
        return NULL;   
    } 
    strcpy(copyname, name);

    int32_t* sizes = (int32_t *) mem;
    sizes[0] = (int32_t) local_size;
    sizes[1] = (int32_t) datastack_size;
    sizes[2] = (int32_t) funcstack_size;
    sizes[3] = (int32_t) program_len;

    program = (operation_double_t *) &sizes[4];

    // Convert the array of data given as the argument prog_array
    // into the array program (an array of operation_double_t).
    data = (simple_operation_double_t *) PyArray_BYTES(prog_array);
    simple_operation_double_t *dp = data;
    for (size_t i = 0; i < program_len; ++i, ++dp) {
        program[i].opcode = dp->opcode;
        switch (program[i].opcode) {
            case PUSH_LOCAL:
            case STORE_LOCAL:
                program[i].operand.index = dp->index;
                break;

            case PUSH_CONSTANT:
                program[i].operand.constant = dp->value;
                break;

            case JUMP:
            case JUMP_IF_FALSE:
            case JUMP_IF_TRUE:
                program[i].operand.jump_destination = dp->index;
                break;

            case PUSH_FUNCTION:
                // The only case handled at the moment: dp->index is the index
                // into the array of C double function data.
                program[i].operand.function = c_double_functions[dp->index];
                break;

            case CALL_FUNCTION:
            case UNARY_POSITIVE:
            case UNARY_NEGATIVE:
            case ADD:
            case SUBTRACT:
            case MULTIPLY:
            case TRUE_DIVIDE:
            case POWER:
            case COMPARE_LT:
            case COMPARE_LE:
            case COMPARE_GT:
            case COMPARE_GE:
            case COMPARE_EQ:
            case COMPARE_NE:
            case RETURN:
            default:
                break;  // No data to copy for these opcodes.
        }
    }

    print_program(program, program_len);

    if (nargs == 1) {
        ufunc_functions[0] = loop_d_d;
    }
    else if (nargs == 2) {
        ufunc_functions[0] = loop_dd_d;
    }
    else {
        // nargs == 3
        ufunc_functions[0] = loop_ddd_d;        
    }
    ufunc_data[0] = mem;

    // Hack: the types array is "\14\14\14\14", which works for 1, 2 or 3 inputs.
    ufunc = PyUFunc_FromFuncAndData(ufunc_functions, ufunc_data,
                                    "\14\14\14\14", 1, nargs, 1,
                                    PyUFunc_None, copyname,
                                    "ufunk-docstring", 0);

    // Set the type of the new ufunc to the version in PyUFunk_Type.
    Py_TYPE(ufunc) = &PyUFunk_Type.ufunc_type;
    return ufunc;
}

static PyMethodDef module_methods[] = {
        {"_ufunkify", _ufunkify, METH_VARARGS,
         "Generate a ufunc from the parsed sequence of ufunkify opcodes.\n"},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_ufunkify",
    .m_doc = "Module that defines the function ufunc_inspector.",
    .m_size = -1,
    .m_methods = module_methods
};

// --------------------------------------------------------------------------
// Methods that will override PyUFunc_Type methods in PyUFunk_Type
// --------------------------------------------------------------------------


static void
ufunk_dealloc(PyUFuncObject *ufunc)
{
    // Free the memory that was allocated when the ufunk was created.
    free(ufunc->data[0]);
    free(ufunc->data);
    free(ufunc->functions);
    free((void *) ufunc->name);

    // Call the parent's tp_dealloc.
    PyUFunk_Type.parent_tp_dealloc((PyObject *)ufunc);
}

static PyObject *
ufunk_repr(PyUFuncObject *ufunc)
{
    return PyUnicode_FromFormat("<ufunk '%s'>", ufunc->name);
}

// --------------------------------------------------------------------------


PyMODINIT_FUNC PyInit__ufunkify(void)
{
    PyObject *m;

    import_array();
    import_umath();

    // Copy PyUFunc_Type to PyUFunk_Type.ufunc_type, so we can create
    // a new type with custom tp_dealloc and tp_repr slots.
    PyUFunk_Type.ufunc_type = PyUFunc_Type;
    PyUFunk_Type.ufunc_type.tp_name = "ufunk";
    // Save the original tp_dealloc and tp_repr.  (Saving tp_repr is
    // an unnecessary formality.  We really only need to save tp_dealloc.)
    PyUFunk_Type.parent_tp_dealloc = PyUFunk_Type.ufunc_type.tp_dealloc;
    PyUFunk_Type.parent_tp_repr    = PyUFunk_Type.ufunc_type.tp_repr;
    // Override tp_dealloc and tp_repr.
    PyUFunk_Type.ufunc_type.tp_dealloc = (destructor) ufunk_dealloc;
    PyUFunk_Type.ufunc_type.tp_repr    = (reprfunc) ufunk_repr;


    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    return m;
}
