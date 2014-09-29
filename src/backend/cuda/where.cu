#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>

#undef _GLIBCXX_USE_INT128
#include <where.hpp>
#include <complex>

namespace cuda
{
    template<typename T>
    Array<uint>* where(const Array<T> &in)
    {
        CUDA_NOT_SUPPORTED();
        return NULL;
    }


#define INSTANTIATE(T)                                  \
    template Array<uint>* where<T>(const Array<T> &in);    \

    INSTANTIATE(float  )
    INSTANTIATE(cfloat )
    INSTANTIATE(double )
    INSTANTIATE(cdouble)
    INSTANTIATE(char   )
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(uchar  )

}
