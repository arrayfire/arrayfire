#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <regions.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
Array<T> * regions(const Array<uchar> &in, const unsigned connectivity)
{
    OPENCL_NOT_SUPPORTED();
}

#define INSTANTIATE(T)\
    template Array<T> * regions<T>(const Array<uchar> &in, const unsigned connectivity);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
