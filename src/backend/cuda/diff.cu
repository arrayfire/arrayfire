#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <diff.hpp>
#include <cassert>

namespace cuda
{
    template<typename T>
    Array<T> *diff1(const Array<T> &in, const int dim)
    {
        assert(1!=1);
        return NULL;
    }

    template<typename T>
    Array<T> *diff2(const Array<T> &in, const int dim)
    {
        assert(1!=1);
        return NULL;
    }

#define INSTANTIATE(T)                                                  \
    template Array<T>* diff1<T>  (const Array<T> &in, const int dim);   \
    template Array<T>* diff2<T>  (const Array<T> &in, const int dim);   \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

}
