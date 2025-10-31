
#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdio.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


typedef double (*doublebinaryfunc)(double, double);


void print_core_dim_flags(npy_uint32 flags)
{
    bool add_bar = false;
    if (flags & UFUNC_CORE_DIM_SIZE_INFERRED) {
        printf("SIZE_INFERRED");
        add_bar = true;
    }
    if (flags & UFUNC_CORE_DIM_CAN_IGNORE) {
        if (add_bar) {
            printf(" | ");
        }
        printf("CAN_IGNORE");
        add_bar = true;
    }
    if (flags & UFUNC_CORE_DIM_MISSING) {
        if (add_bar) {
            printf(" | ");
        }
        printf("MISSING");
    }
}


static char
get_typechar_from_typenum(int typenum)
{
    char c;

    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr != NULL) {
        c = descr->type;
        Py_DECREF(descr);
    }
    else {
        // Something to indicate an error occurred.
        c = '!';
    }
    return c;
}

static char null_name[] = "<NULL>";
static char unknown_name[] = "<unknown>";

static const char *
get_typename_from_typenum(int typenum)
{
    const char *name;

    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr != NULL) {
        name = descr->typeobj->tp_name;
        Py_DECREF(descr);
        if (name == NULL) {
            name = null_name;
        }
    }
    else {
        // Something to indicate an error occurred.
        name = unknown_name;
    }
    return name;
}


static PyObject *
ufunc_inspector(PyObject *self, PyObject *arg)
{
    PyUFuncObject *ufunc;

    if (!PyObject_TypeCheck(arg, &PyUFunc_Type)) {
        printf("Not a ufunc.\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ufunc = (PyUFuncObject *) arg;

    if (ufunc->core_enabled) {
        printf("'%s' is a gufunc with signature '%s'.\n", ufunc->name, ufunc->core_signature);
        printf("nin = %d, nout = %d\n", ufunc->nin, ufunc->nout);
        printf("...core_num_dim_ix  (number of distinct names in sig) = %d\n", ufunc->core_num_dim_ix);
        printf("...core_dim_sizes                  (-1 if not frozen) = [");
        for (int i = 0; i < ufunc->core_num_dim_ix; ++i) {
            printf("%lld", (long long int) ufunc->core_dim_sizes[i]);
            if (i == ufunc->core_num_dim_ix-1) {
                printf("]\n");
            }
            else {
                printf(", ");
            }
        }
        printf("...core_dim_flags             (UFUNC_CORE_DIM* flags) = [");
        for (int i = 0; i < ufunc->core_num_dim_ix; ++i) {
            print_core_dim_flags(ufunc->core_dim_flags[i]);
            //printf("%u", ufunc->core_dim_flags[i]);
            if (i == ufunc->core_num_dim_ix-1) {
                printf("]\n");
            }
            else {
                printf(", ");
            }
        }
        printf("...core_num_dims    (number of core dims in each arg) = [");
        for (int i = 0; i < ufunc->nargs; ++i) {
            printf("%d", ufunc->core_num_dims[i]);
            if (i == ufunc->nargs-1) {
                printf("]\n");
            }
            else {
                printf(", ");
            }
        }
        printf("...core_dim_ixs      (dimension indices for each arg) =");
        for (int i = 0; i < ufunc->nargs; ++i) {
            int offset = ufunc->core_offsets[i];
            printf(" [");
            for (int j = offset; j < offset + ufunc->core_num_dims[i]; ++j) {
                printf("%d", ufunc->core_dim_ixs[j]);
                if (j != offset + ufunc->core_num_dims[i]-1) {
                    printf(", ");
                }
            }
            printf("]");
        }
        printf("\n");
        printf("...core_offsets    (pos. of 1st core dim of each arg) = [");
        for (int i = 0; i < ufunc->nargs; ++i) {
            printf("%d", ufunc->core_offsets[i]);
            if (i == ufunc->nargs-1) {
                printf("]\n");
            }
            else {
                printf(", ");
            }
        }
        printf("...op_flags (flags for each op when called by nditer) = [");
        for (int i = 0; i < ufunc->nargs; ++i) {
            printf("%u", ufunc->op_flags[i]);
            if (i == ufunc->nargs-1) {
                printf("]\n");
            }
            else {
                printf(", ");
            }
        }
    }
    else {
        printf("'%s' is a ufunc.\n", ufunc->name);
        printf("nin = %d, nout = %d\n", ufunc->nin, ufunc->nout);
    }
    printf("ntypes = %d\n", ufunc->ntypes);

    if (ufunc->ntypes > 0 && ufunc->nin > 0 && ufunc->nin < 5 && ufunc->nout == 1) {
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
            printf("(%c->%c)  ", get_typechar_from_typenum(ufunc->types[2*i]),
                                 get_typechar_from_typenum(ufunc->types[2*i+1]));
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
            else CHECKFOR(O_O)
            else CHECKFOR(O_O_method)
            else CHECKFOR(On_Om)
            else {
                printf("not generic (or not in the checked generics)\n");
            }
        }
        else if (ufunc->nin == 2 && ufunc->nout == 1) {
            printf("%3d: (%3d, %3d) -> %3d  ",
                   i, ufunc->types[3*i], ufunc->types[3*i+1], ufunc->types[3*i+2]);
            printf("(%c%c->%c)  ", get_typechar_from_typenum(ufunc->types[3*i]),
                                   get_typechar_from_typenum(ufunc->types[3*i+1]),
                                   get_typechar_from_typenum(ufunc->types[3*i+2]));
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
            printf("(%c%c%c->%c)  ", get_typechar_from_typenum(ufunc->types[4*i]),
                                     get_typechar_from_typenum(ufunc->types[4*i+1]),
                                     get_typechar_from_typenum(ufunc->types[4*i+2]),
                                     get_typechar_from_typenum(ufunc->types[4*i+3]));
            printf("\n");
        }
        else if (ufunc->nin == 4 && ufunc->nout == 1) {
            printf("%3d: (%3d, %3d, %3d, %3d) -> %3d  ", i,
                   ufunc->types[5*i], ufunc->types[5*i+1],
                   ufunc->types[5*i+2], ufunc->types[5*i+3],
                   ufunc->types[5*i+4]);
            printf("(%c%c%c%c->%c)  ", get_typechar_from_typenum(ufunc->types[5*i]),
                                       get_typechar_from_typenum(ufunc->types[5*i+1]),
                                       get_typechar_from_typenum(ufunc->types[5*i+2]),
                                       get_typechar_from_typenum(ufunc->types[5*i+3]),
                                       get_typechar_from_typenum(ufunc->types[5*i+4]));
            printf("\n");
        }
    }

    if (ufunc->userloops != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        printf("Registered user loops:\n");
        while (PyDict_Next(ufunc->userloops, &pos, &key, &value)) {
            PyUFunc_Loop1d *current;
            // value is a PyCapsule
            printf("  typenum = ");
            PyObject_Print(key, stdout, 0);
            long num_type = PyLong_AsLong(key);
            if (!PyErr_Occurred()) {
                printf("; type name is '%s'", get_typename_from_typenum(num_type));
            }
            printf("\n");
            current = (PyUFunc_Loop1d *) PyCapsule_GetPointer(value, NULL);
            while (current != NULL) {
                printf("      arg_types:  in:");
                for (int i = 0; i < ufunc->nin + ufunc->nout; ++i) {
                    if (i == ufunc->nin) {
                        printf("   out:");
                    }
                    printf(" %d", current->arg_types[i]);
                }
                printf("\n");
                current = current->next;
            }
        }
    }
/*
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
*/

/*
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
*/

    Py_INCREF(Py_None);
    return Py_None;
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

    if (PyArray_ImportNumPyAPI() < 0) {
        return NULL;
    }
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    return m;
}
