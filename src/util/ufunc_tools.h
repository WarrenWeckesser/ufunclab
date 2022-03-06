
// This was originally going to be a small C *library* of
// utility functions to be used in the extension modules,
// but I ran into some problems getting the library to
// compile, link and run without seg. faults with the
// extension modules, so it is a header file instead. Ugh.

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


#ifdef __cplusplus
extern "C" {
#endif

// Compute nin and nout from the signature.
// This counts the number of opening parenthesis
// characters ('(') before and after the '-' character.
// This function does not validate the signature. E.g.
// signature = "($#-(()((" will return 1 and 4 for nin
// and nout, respectively.
//
static void
get_nin_nout(const char *signature, int *nin, int *nout)
{
    int *pcount = nin;

    *nin = 0;
    *nout = 0;
    for (const char *p = signature; *p; ++p) {
        if (*p == '-') {
            pcount = nout;
        }
        else if (*p == '(') {
            ++*pcount;
        }
    }
}

//
// Create the gufunc from the given arguments and add it to the module.
// Return 0 on success, -1 on error.
//

static int
ul_define_gufunc(PyObject *module, const char *name, const char *doc,
                 const char *signature, int ntypes,
                 PyUFuncGenericFunction funcs[], void *data[], char *types)
{
    PyUFuncObject *gufunc;
    int nin, nout;
    int status;

    get_nin_nout(signature, &nin, &nout);

    // Create the ufunc.
    gufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                        funcs, data, types, ntypes, nin, nout,
                        PyUFunc_None, name, doc, 0, signature);
    if (gufunc == NULL) {
        return -1;
    }
    status = PyModule_AddObject(module, name, (PyObject *) gufunc);
    if (status == -1) {
        Py_DECREF(gufunc);
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

