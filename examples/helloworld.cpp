#include <arrayfire.h>
#include <iostream>

using namespace af;

int main(int argc, char *argv[])
{
    array A = randu(10);
    af_print(A);
    return 0;
}
