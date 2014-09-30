#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <regions.hpp>
#include <err_cpu.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
Array<T> * regions(const Array<uchar> &in, const unsigned connectivity)
{
    CPU_NOT_SUPPORTED();
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
