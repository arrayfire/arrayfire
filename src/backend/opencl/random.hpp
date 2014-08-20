#include <af/array.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    Array<T>* randu(const af::dim4 &dims);

    template<typename T>
    Array<T>* randn(const af::dim4 &dims);
}
