//
// A simple C main program to exercise the code in debye1_generated.c.
// Compile with (for example):
//     $ gcc check_debye1_generated.c debye1_generated.c -o check_debye1_generated
// Pass x values on the command line, e.g. on Linux:
//     $ ./check_debye1_generated 0 0.25 5.5 -12.0
// That will print:
//     0.00000000000000000e+00  1.00000000000000000e+00
//     2.50000000000000000e-01  9.39235027193614513e-01
//     5.50000000000000000e+00  2.94239966231542471e-01
//    -1.20000000000000000e+01  6.13707118265430740e+00
//

#include <stdio.h>
#include <stdlib.h>

#include "debye1_generated.h"


int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        double x = strtod(argv[i], NULL);
        double y = debye1(x);
        printf("%25.17e  %.17e\n", x, y);
    }
}
