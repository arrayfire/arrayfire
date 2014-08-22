#pragma once

namespace cuda
{
namespace kernel
{

    template<typename T>
    void transpose( T * out, const T * in, const dim_type ndims, const dim_type * const dims, const dim_type * const strides);

}
}
