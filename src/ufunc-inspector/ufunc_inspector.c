#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


typedef double (*doublebinaryfunc)(double, double);


static PyObject *
ufunc_inspector(PyObject *self, PyObject *arg)
{
    PyUFuncObject *ufunc;
    PyObject *return_value = NULL;

    if (!PyObject_TypeCheck(arg, &PyUFunc_Type)) {
        printf("Not a ufunc.\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ufunc = (PyUFuncObject *) arg;

    printf("'%s' is a ufunc.\n", ufunc->name);

    printf("nin = %d, nout = %d, ntypes = %d\n", ufunc->nin, ufunc->nout, ufunc->ntypes);

    if (ufunc->ntypes > 0) {
        printf("loop types:\n");
    }
    for (int i = 0; i < ufunc->ntypes; ++i) {
        // XXX I'm not sure all the existing generic loops are checked for...
        #define CHECKFOR(sig)                               \
            if (ufunc->functions[i] == PyUFunc_##sig) {     \
                printf("PyUFunc_" #sig "\n");               \
            }

        if (ufunc->nin == 1 && ufunc->nout == 1) {
            printf("%3d:  %3d -> %3d  ",
                   i, ufunc->types[2*i], ufunc->types[2*i+1]);
            printf("(%c->%c)  ",
                   PyArray_DescrFromType(ufunc->types[2*i])->type,
                   PyArray_DescrFromType(ufunc->types[2*i+1])->type);
            CHECKFOR(e_e)
            else CHECKFOR(e_e_As_f_f)
            else CHECKFOR(e_e_As_d_d)
            else CHECKFOR(f_f)
            else CHECKFOR(f_f_As_d_d)
            else CHECKFOR(d_d)
            else CHECKFOR(g_g)
            else CHECKFOR(F_F)
            else CHECKFOR(F_F_As_D_D)
            else CHECKFOR(D_D)
            else CHECKFOR(G_G)
            else CHECKFOR(O_O_method)
            else CHECKFOR(On_Om)
            else {
                printf("not generic (or not in the checked generics)\n");
            }
        }
        else if (ufunc->nin == 2 && ufunc->nout == 1) {
            printf("%3d: (%3d, %3d) -> %3d  ",
                   i, ufunc->types[3*i], ufunc->types[3*i+1], ufunc->types[3*i+2]);
            printf("(%c%c->%c)  ",
                   PyArray_DescrFromType(ufunc->types[3*i])->type,
                   PyArray_DescrFromType(ufunc->types[3*i+1])->type,
                   PyArray_DescrFromType(ufunc->types[3*i+2])->type);
            CHECKFOR(ee_e)
            else CHECKFOR(ee_e_As_ff_f)
            else CHECKFOR(ee_e_As_dd_d)
            else CHECKFOR(ff_f)
            else CHECKFOR(ff_f_As_dd_d)
            else CHECKFOR(dd_d)
            else CHECKFOR(gg_g)
            else CHECKFOR(FF_F)
            else CHECKFOR(FF_F_As_DD_D)
            else CHECKFOR(DD_D)
            else CHECKFOR(GG_G)
            else CHECKFOR(OO_O)
            else CHECKFOR(OO_O_method)
            else {
                printf("not generic (or not in the checked generics)\n");
            }
        }
        else if (ufunc->nin == 3 && ufunc->nout == 1) {
            printf("%3d: (%3d, %3d, %3d) -> %3d  ", i,
                   ufunc->types[4*i], ufunc->types[4*i+1], ufunc->types[4*i+2],
                   ufunc->types[4*i+3]);
            printf("(%c%c%c->%c)  ",
                   PyArray_DescrFromType(ufunc->types[4*i])->type,
                   PyArray_DescrFromType(ufunc->types[4*i+1])->type,
                   PyArray_DescrFromType(ufunc->types[4*i+2])->type,
                   PyArray_DescrFromType(ufunc->types[4*i+3])->type);
            printf("\n");
        }
    }

    if (ufunc->userloops != NULL) {
        printf("userloops is not NULL\n");
        return_value = ufunc->userloops;
    }
    // Some experiments...
    if (ufunc->nin == 2 && ufunc->nout == 1) {
        // look for dd->d
        int k = 0;
        for (int i = 0; i < ufunc->ntypes; ++i) {
            if (ufunc->types[k] == NPY_DOUBLE &&
                    ufunc->types[k+1] == NPY_DOUBLE &&
                    ufunc->types[k+2] == NPY_DOUBLE) {
                printf("Found type dd->d\n");
                printf("Let's try calling the inner loop function.\n");
                double x = 3.0;
                double y = 4.0;
                double z;
                double *arg[] = {&x, &y, &z};
                npy_intp n = 1;
                npy_intp steps[] = {sizeof(x), sizeof(y), sizeof(z)};
                (ufunc->functions[i])((char **) &arg, &n, steps, ufunc->data[i]);
                printf("x = %10.6f, y = %10.6f, z = %10.6f\n", x, y, z);
                break;
            }
            k += 3;
        }
    }

    // Another experiment: look for PyUFunc_dd_d
    for (int i = 0; i < ufunc->ntypes; ++i) {
        if (ufunc->functions[i] == PyUFunc_dd_d) {
            printf("\n");
            printf("Found PyUFunc_dd_d\n");
            printf("Let's try calling ufunc->data[i]\n");
            double x = 3.0;
            double y = 4.0;
            double z = ((doublebinaryfunc) (ufunc->data[i]))(x, y);
            printf("x = %10.6f, y = %10.6f, z = %10.6f\n", x, y, z);
        }
    }

    if (return_value == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    Py_INCREF(return_value);
    return return_value;
}

static PyMethodDef module_methods[] = {
        {"ufunc_inspector", ufunc_inspector, METH_O,
         "ufunc_inspector(ufunc)\n\nPrint information about a ufunc.\n"},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_ufunc_inspector",
    .m_doc = "Module that defines the function ufunc_inspector.",
    .m_size = -1,
    .m_methods = module_methods
};


PyMODINIT_FUNC PyInit__ufunc_inspector(void)
{
    PyObject *m;

    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    return m;
}
