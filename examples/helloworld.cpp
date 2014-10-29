#include <arrayfire.h>
#include <iostream>

using namespace af;

int main(int argc, char *argv[])
{
    af::info();
    array A = randu(5, 4);
    af_print(A);

    array B = sum(A, 0);
    af_print(B);

    array C = diff1(A);
    af_print(C);

    float h_array[] = {5, 3, 6, 8, 0, 2, 4, 7, 9, 1};
    array D(10, h_array, af::afHost);
    af_print(D);

    array E, F;
    sort_index(E, F, D);
    af_print(E);
    af_print(F);

    return 0;
}
