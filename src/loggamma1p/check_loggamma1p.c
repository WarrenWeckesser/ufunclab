#include <stdio.h>
#include <stdlib.h>

extern double loggamma1p(double x);

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        double x = strtod(argv[i], NULL);
        double y = loggamma1p(x);
        printf("%24.16e %24.16e\n", x, y);
    }
}