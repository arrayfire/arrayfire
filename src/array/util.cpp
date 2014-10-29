#include <af/array.h>
#include <af/util.h>
#include <iostream>
#include "error.hpp"

using namespace std;

namespace af
{
    void print(const char *exp, const array &arr)
    {
        std::cout << exp << std::endl;
        AF_THROW(af_print_array(arr.get()));
        return;
    }
}
