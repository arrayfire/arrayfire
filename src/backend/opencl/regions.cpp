#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <regions.hpp>
#include <kernel/regions.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
Array<T> * regions(const Array<uchar> &in, const unsigned connectivity)
{
    ARG_ASSERT(2, (connectivity==4 || connectivity==8));

    const af::dim4 dims = in.dims();

    Array<T> * out  = createEmptyArray<T>(dims);

    switch(connectivity) {
        case 4:
            kernel::regions<T, false, 2>(*out, in);
            break;
        case 8:
            kernel::regions<T, true,  2>(*out, in);
            break;
    }

    return out;
}

#define INSTANTIATE(T)                                                                      \
    template Array<T> * regions<T>(const Array<uchar> &in, const unsigned connectivity);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
