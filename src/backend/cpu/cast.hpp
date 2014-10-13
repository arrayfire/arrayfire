#pragma once
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <complex>
#include <err_cpu.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>

namespace cpu
{

template<typename To, typename Ti>
Array<To>* cast(const Array<Ti> &in)
{
    CPU_NOT_SUPPORTED();
}

}
