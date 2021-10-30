
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>

// Only need stdio.h while debugging.
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>


typedef struct gendot_data_t {
    PyUFuncGenericFunction *prodfunc_loop;
    void *prodfunc_loop_data;
    PyUFuncGenericFunction *sumfunc_loop;
    void *sumfunc_loop_data;
    npy_intp sumfunc_loop_itemsize;
    npy_intp sumfunc_nin;
    npy_intp sumfunc_has_identity; // actual values are only 0 or 1.
    // sumfunc_identity_buffer is large enough to hold an instance
    // of np.complex256
    uint8_t sumfunc_identity_buffer[32];
} gendot_data_t;

#define PRODFUNC_LOOP(p)            (((gendot_data_t *)(p))->prodfunc_loop)
#define PRODFUNC_LOOP_DATA(p)       (((gendot_data_t *)(p))->prodfunc_loop_data)
#define SUMFUNC_LOOP(p)             (((gendot_data_t *)(p))->sumfunc_loop)
#define SUMFUNC_LOOP_DATA(p)        (((gendot_data_t *)(p))->sumfunc_loop_data)
#define SUMFUNC_LOOP_ITEMSIZE(p)    (((gendot_data_t *)(p))->sumfunc_loop_itemsize)
#define SUMFUNC_NIN(p)              (((gendot_data_t *)(p))->sumfunc_nin)
#define SUMFUNC_HAS_IDENTITY(p)     (((gendot_data_t *)(p))->sumfunc_has_identity)
#define SUMFUNC_IDENTITY_BUFFER(p)  (((gendot_data_t *)(p))->sumfunc_identity_buffer)


typedef struct funct_index_pair {
    uint8_t prod_index;
    uint8_t sum_index;
} func_index_pair;


//
// result must point to an instance of the output data type.
// loop_data must be the pointer that is passed to the data argument
// of loop_function when it is called.
//
static void
reduce(PyUFuncGenericFunction *loop_function, void *loop_data,
       char *data, npy_intp n, npy_intp stride, npy_intp itemsize,
       char *result)
{
    char *loop_args[3];
    npy_intp loop_dimensions[1];
    npy_intp loop_steps[3];

    if (n <= 0) {
        return;
    }
    memcpy(result, data, itemsize);
    if (n == 1) {
        return;
    }
    loop_args[0] = result;
    loop_args[2] = result;
    data += stride;
    loop_dimensions[0] = 1;
    // XXX The values in loop_steps shouldn't matter, since
    //     loop_dimensions[0] is 1 in each call of loop_functions.
    loop_steps[0] = stride;
    loop_steps[1] = stride;
    loop_steps[2] = stride;
    for (int k = 1; k < n; ++k, data += stride) {
        loop_args[1] = data;
        ((PyUFuncGenericFunction)loop_function)(loop_args, loop_dimensions, loop_steps, loop_data);
    }
}


static void
gendot_loop(char **args, const npy_intp *dimensions,
            const npy_intp* steps, void* data)
{
    // dimensions[0]: Number of input arrays
    // dimensions[1]: Length of first two input arrays.
    // steps[0]:  x array outer step
    // steps[1]:  y array outer step
    // steps[2]:  out array outer step
    // steps[3]:  inner x array step
    // steps[4]:  inner y array step

    char *px = args[0];
    char *py = args[1];
    char *pout = args[2];
    npy_intp nloops = dimensions[0];

    char *prodfunc_args[3];
    npy_intp prodfunc_dimensions[3];
    npy_intp prodfunc_steps[3];

    size_t itemsize = SUMFUNC_LOOP_ITEMSIZE(data);

    if (dimensions[1] == 0) {
        if (SUMFUNC_HAS_IDENTITY(data)) {
            // Attempting to apply the reduction to empty sequences.
            // Return the identity element.
            for (int j = 0; j < nloops; ++j, pout += steps[2]) {
                memcpy(pout, SUMFUNC_IDENTITY_BUFFER(data), itemsize);
            }
            return;
        }
        else {
            // Applying the reduction to an empty sequence is not
            // allowed if there is no identity element.
            NPY_ALLOW_C_API_DEF
            NPY_ALLOW_C_API
                PyErr_SetString(PyExc_ValueError,
                    "zero-size array to reduction operation with no identity");
            NPY_DISABLE_C_API
            return;
        }
    }

    char *tmp = malloc(dimensions[1] * itemsize);
    if (tmp == NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
            PyErr_Format(PyExc_MemoryError,
                "Unable to allocate %ld bytes (%ld items, each with size %ld) "
                "for intermediate calculation",
                dimensions[1] * itemsize, dimensions[1], itemsize);
        NPY_DISABLE_C_API
        return;
    }

    PyUFuncGenericFunction *prodfunc_loop = PRODFUNC_LOOP(data);
    void * prodfunc_data = PRODFUNC_LOOP_DATA(data);

    // prodfunc_args[0] and prodfunc_args[1] are updated dynamically
    // in the loop below.
    prodfunc_args[2] = tmp;
    for (int k = 0; k < 3; ++k) {
        prodfunc_dimensions[k] = dimensions[1];
    }
    prodfunc_steps[0] = steps[3];
    prodfunc_steps[1] = steps[4];
    prodfunc_steps[2] = itemsize;

    PyUFuncGenericFunction * sumfunc_loop = SUMFUNC_LOOP(data);
    void *sumfunc_loop_data = SUMFUNC_LOOP_DATA(data);

    if (SUMFUNC_NIN(data) == 2) {
        // sumfunc is an element-wise ufunc with 2 inputs and 1 output.

        for (int j = 0; j < nloops; ++j, px += steps[0],
                                         py += steps[1],
                                         pout += steps[2]) {
            prodfunc_args[0] = px;
            prodfunc_args[1] = py;
            ((PyUFuncGenericFunction)prodfunc_loop)(prodfunc_args, prodfunc_dimensions,
                                                    prodfunc_steps, prodfunc_data);
            reduce(sumfunc_loop, sumfunc_loop_data,
                   tmp, dimensions[1], itemsize, itemsize, pout);
        }
    }
    else {
        // sumfunc is a gufunc with signature (n)->().  Call its loop
        // function directly to compute the output value.

        // sumfunc_args[1] is updated in each iteration of the loop below.
        char *sumfunc_args[2] = {tmp, NULL};
        npy_intp sumfunc_dimensions[2] = {1, dimensions[1]};
        npy_intp sumfunc_steps[3] = {0, 0, itemsize};
        // Note that sumfunc_steps[0] and sumfunc_steps[1] will not actually be
        // used in sumfunc_loop, because sumfunc_dimensions[0] is 1.

        for (int j = 0; j < nloops; ++j, px += steps[0],
                                         py += steps[1],
                                         pout += steps[2]) {
            prodfunc_args[0] = px;
            prodfunc_args[1] = py;
            ((PyUFuncGenericFunction)prodfunc_loop)(prodfunc_args, prodfunc_dimensions,
                                                    prodfunc_steps, prodfunc_data);
            sumfunc_args[1] = pout;
            ((PyUFuncGenericFunction)sumfunc_loop)(sumfunc_args, sumfunc_dimensions,
                                                   sumfunc_steps, sumfunc_loop_data);
        }
    }
    free(tmp);
}


//
// This is the function that implements ufunclab.gendot.
// It generates the gufunc that is the composition of the
// two given ufuncs.
//
static PyObject *
gendot(PyObject *self, PyObject *args, PyObject *kwargs)
{
    char *name;
    char *doc;
    PyUFuncObject *prodfunc = NULL;
    PyUFuncObject *sumfunc = NULL;
    int sumfunc_has_identity;
    PyObject *sumfunc_identity_array = NULL;
    PyObject *loop_indices = NULL;
    PyObject *typecodes = NULL;
    PyObject *itemsizes = NULL;
    static char *kwlist[] = {"name", "doc", "prodfunc", "sumfunc",
                             "sumfunc_has_identity",
                             "sumfunc_identity_array",
                             "loop_indices", "typecodes", "itemsizes", NULL};
    // The type checks are included here, but in fact, the code will assume
    // that the inputs have all been validated by the calling code.  Invalid
    // arguments (e.g. arrays with the wrong size or wrong data type) will
    // almost certainly cause the program to crash.
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssO!O!pO!O!O!O!:gendot", kwlist,
                                     &name,
                                     &doc,
                                     &PyUFunc_Type, &prodfunc,
                                     &PyUFunc_Type, &sumfunc,
                                     &sumfunc_has_identity,
                                     &PyArray_Type, &sumfunc_identity_array,
                                     &PyArray_Type, &loop_indices,
                                     &PyArray_Type, &typecodes,
                                     &PyArray_Type, &itemsizes)) {
        return NULL;
    }

    // loop_indices is an array with shape (nloops, 2).
    // func_index_pairs is a 1-d arry of func_index_pair structs that is
    // a view of the 2-d array in loop_indices.
    func_index_pair *func_index_pairs =
            (func_index_pair *) PyArray_DATA((PyArrayObject *)loop_indices);

    npy_intp nloops = PyArray_DIMS((PyArrayObject *)loop_indices)[0];
    npy_intp *gendot_itemsizes = PyArray_DATA((PyArrayObject *)itemsizes);

    uint8_t *sumfunc_identity_data =
            PyArray_DATA((PyArrayObject *)sumfunc_identity_array);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Allocate arrays for PyUFunc_FromFuncAndDataAndSignature.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // We have to allocate a single block of memory, and then assign the
    // arrays to offsets within that block.  By having a single block,
    // we can store the pointer to the block in the ptr field of the ufunc
    // object. The ufunc will free the block when its dealloc method
    // is called.

    // This code is probably more verbose than necessary...
    size_t sizeof_gendot_funcs     = nloops * sizeof(PyUFuncGenericFunction *);
    size_t sizeof_gendot_data_ptrs = nloops * sizeof(void *);
    size_t sizeof_gendot_data      = nloops * sizeof(gendot_data_t);
    size_t sizeof_gendot_typecodes = nloops * 3*sizeof(uint8_t);

    // The total includes the memory required for copies of the name and
    // doc strings.
    size_t total_memory_required = (sizeof_gendot_funcs + sizeof_gendot_data_ptrs
                                    + sizeof_gendot_data + sizeof_gendot_typecodes
                                    + (strlen(name) + 1) + (strlen(doc) + 1));

    // We must use PyArray_malloc, because this memory will be freed
    // by the gufunc's dealloc method, and that code uses PyArray_free.
    char *mem = PyArray_malloc(total_memory_required);
    if (mem == NULL) {
        return PyErr_NoMemory();
    }
    // gendot_funcs and gendot_data_ptrs are arrays of pointers, and all the
    // fields in gendot_data_t are pointer-sized, so the alignment of the
    // gendot_data_ptrs and gendot_data pointers should be OK.
    PyUFuncGenericFunction *gendot_funcs = (PyUFuncGenericFunction *) mem;
    void **gendot_data_ptrs = (void *) (mem
                                       + sizeof_gendot_funcs);
    gendot_data_t *gendot_data = (gendot_data_t *) (mem
                                                    + sizeof_gendot_funcs
                                                    + sizeof_gendot_data_ptrs);
    char *gendot_typecodes = (char *) (mem
                                       + sizeof_gendot_funcs
                                       + sizeof_gendot_data_ptrs
                                       + sizeof_gendot_data);
    char *gendot_name = (char *) (mem
                                  + sizeof_gendot_funcs
                                  + sizeof_gendot_data_ptrs
                                  + sizeof_gendot_data
                                  + sizeof_gendot_typecodes);
    char *gendot_doc = (char *) (mem
                                 + sizeof_gendot_funcs
                                 + sizeof_gendot_data_ptrs
                                 + sizeof_gendot_data
                                 + sizeof_gendot_typecodes
                                 + (strlen(name) + 1));

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Fill in the allocated arrays.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // We just allocated the appropriate amount of space for these arrays,
    // so it should be safe to use memcpy and strcpy.
    memcpy(gendot_typecodes, PyArray_DATA((PyArrayObject *)typecodes),
           sizeof_gendot_typecodes);
    strcpy(gendot_name, name);
    strcpy(gendot_doc, doc);

    for (int i = 0; i < nloops; ++i) {
        gendot_funcs[i] = (PyUFuncGenericFunction) &gendot_loop;

        npy_intp prod_index = func_index_pairs[i].prod_index;
        npy_intp sum_index = func_index_pairs[i].sum_index;
        gendot_data[i].prodfunc_loop = (PyUFuncGenericFunction *) prodfunc->functions[prod_index];
        gendot_data[i].prodfunc_loop_data = prodfunc->data[prod_index];
        gendot_data[i].sumfunc_loop = (PyUFuncGenericFunction *) sumfunc->functions[sum_index];
        gendot_data[i].sumfunc_loop_data = sumfunc->data[sum_index];
        gendot_data[i].sumfunc_nin = (npy_intp) sumfunc->nin;
        gendot_data[i].sumfunc_loop_itemsize = gendot_itemsizes[i];
        gendot_data[i].sumfunc_has_identity = (npy_intp) sumfunc_has_identity;
        memcpy(gendot_data[i].sumfunc_identity_buffer,
               sumfunc_identity_data + 32*i,  // FIXME: hard-coded constant 32 :(
               gendot_itemsizes[i]);

        gendot_data_ptrs[i] = &gendot_data[i];
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Create the gufunc.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // Named constants for readability.
    int nin = 2;
    int nout = 1;
    int unused = 0;
    PyUFuncObject * gufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                        gendot_funcs,      // array with length nloops
                        gendot_data_ptrs,  // array with length nloops
                        gendot_typecodes,  // "2-d" array with shape (nloops, 3)
                        nloops, nin, nout,
                        PyUFunc_None, gendot_name, gendot_doc, unused,
                        "(i),(i)->()");
    if (gufunc == NULL) {
        PyArray_free(mem);
        return NULL;
    }
    // This memory is freed in the gufunc's dealloc method.
    gufunc->ptr = mem;
    return (PyObject *) gufunc;
}


static char gendot_docstring[] =
"_gendot(name, doc, prodfunc, sumfunc, sumfunc_identity_array, "
"loop_indices, typecodes, itemsizes)\n"
"\n"
"*** This is not a public function! Use at your own risk! ***\n"
"\n"
"Create a gufunc that computes the generalized dot product.\n"
"\n"
"The function creates a new gufunc (with signature (n),(n)->())\n"
"that is the composition of the two ufunc prodfunc and sumfunc.\n"
"The input ufuncs must each have 2 inputs and 1 output.\n"
"\n"
"The input parameters are not validated.  Passing invalid parameters\n"
"will crash the Python interpreter.\n"
"\n"
"See the Python wrapper function for full details.\n"
"";


static PyMethodDef gendot_methods[] = {
    {"_gendot", (PyCFunction)(void(*)(void)) gendot, METH_VARARGS | METH_KEYWORDS,
     gendot_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gendotmodule = {
    PyModuleDef_HEAD_INIT,
    "_gendot",
    "The _gendot module defines the function _gendot.\n\n"
    "The function _gendot creates a new gufunc (with signature (n),(n)->())\n"
    "that is the composition of two ufuncs.\n",
    -1,
    gendot_methods
};


PyMODINIT_FUNC
PyInit__gendot(void)
{
    PyObject *module;

    module = PyModule_Create(&gendotmodule);
    if (module == NULL) {
        return NULL;
    }

    // Required to access the NumPy C API.
    import_array();
    import_ufunc();

    return module;
}
