from os import path


# TODO: Complete these dictionaries

typechar_to_ctype = dict(
    f='float',
    d='double',
    g='long double',
    F='std::complex<float>',
    D='std::complex<double>',
    G='std::complex<long double>',
)


typechar_to_npy_type = {
    '?': 'NPY_BOOL',
    'b': 'NPY_BYTE',
    'h': 'NPY_SHORT',
    'i': 'NPY_INT',
    'l': 'NPY_LONG',
    'q': 'NPY_LONGLONG',
    'p': 'NPY_INTP',
    'B': 'NPY_UBYTE',
    'H': 'NPY_USHORT',
    'I': 'NPY_UINT',
    'L': 'NPY_ULONG',
    'Q': 'NPY_ULONGLONG',
    'f': 'NPY_FLOAT',
    'd': 'NPY_DOUBLE',
    'g': 'NPY_LONGDOUBLE',
    'F': 'NPY_CFLOAT',
    'D': 'NPY_CDOUBLE',
    'G': 'NPY_CLONGDOUBLE',
    'M': 'NPY_DATETIME',
    'm': 'NPY_TIMEDELTA',
    'O': 'NPY_OBJECT',
    'p': 'NPY_INTP',
    'P': 'NPY_UINTP'
}


def typechar_to_npy_ctype(c):
    if c != 'O':
        ctype = typechar_to_npy_type[c].lower()
    else:
        ctype = "PyObject *"
    return ctype


def typesig_to_ext(typesig):
    """
    typesig must be a ufunc type signature, e.g. 'fff->f'.
    This function just replaces '->' with '_'.  So for input
    'fff->f', the return value is 'fff_f'.
    """
    return typesig.replace('->', '_')


def header_to_concrete_filenames(header):
    """
    Generate the "concrete" header and source filenames from the
    given header.

    Examples
    --------
    >>> header_to_concrete_filenames('foo.h')
    ('foo_concrete.h', 'foo_concrete.cxx')

    """
    root, ext = path.splitext(header)
    ext = ext.lstrip(path.extsep)
    if ext not in ['h', 'hh', 'hpp', 'h++']:
        raise RuntimeError("unexpectd file extension for header "
                           f"file '{header}'")

    root_concrete = root + '_concrete'
    cxxfilename = root_concrete + path.extsep + 'cxx'
    cxxheader = root_concrete + path.extsep + 'h'
    return cxxheader, cxxfilename
