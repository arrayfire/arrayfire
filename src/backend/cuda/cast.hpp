#pragma once
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <complex>
#include <err_cuda.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>

namespace cuda
{

template<typename To, typename Ti>
Array<To>* cast(const Array<Ti> &in)
{
    CUDA_NOT_SUPPORTED();
}

}
