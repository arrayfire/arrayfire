#define __CL_ENABLE_EXCEPTIONS
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <histogram.hpp>
#include <kernel/histogram.hpp>
#include <stdexcept>
#include <iostream>
#include <errorcodes.hpp>

using af::dim4;

namespace opencl
{

template<typename inType, typename outType>
Array<outType> * histogram(const Array<inType> &in, const unsigned &nbins, const double &minval, const double &maxval)
{
    if (nbins>kernel::MAX_BINS)
        throw std::runtime_error("@histogram: maximum bins exceeded.");

    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();
    dim4 outDims        = dim4(nbins, 1, dims[2], dims[3]);
    Array<outType>* out = createValueArray<outType>(outDims, outType(0));
    const dim4 ostrides = out->strides();

    // create an array to hold min and max values for
    // batch operation handling, this will reduce
    // number of concurrent reads to one single memory location
    cfloat* h_minmax = new cfloat[dims[2]];

    for(dim_type k=0; k<dims[2]; ++k) {
        h_minmax[k].s[0] = minval;
        h_minmax[k].s[1] = maxval;
    }

    dim4 minmax_dims(dims[2]*2);
    Array<cfloat>* minmax = createDataArray<cfloat>(minmax_dims, h_minmax);

    // cleanup the host memory used
    delete[] h_minmax;

    kernel::HistParams params;

    params.offset = in.getOffset();
    for(dim_type i=0; i<4; ++i) {
        params.idims[i]    = dims[i];
        params.istrides[i] = istrides[i];
        params.ostrides[i] = ostrides[i];
    }

    kernel::histogram<inType, outType>(out->get(), in.get(), minmax->get(), params, nbins);

    // destroy the minmax array
    destroyArray(*minmax);

    return out;
}

#define INSTANTIATE(in_t,out_t)\
template Array<out_t> * histogram(const Array<in_t> &in, const unsigned &nbins, const double &minval, const double &maxval);

INSTANTIATE(float,uint)
INSTANTIATE(double,uint)
INSTANTIATE(char,uint)
INSTANTIATE(int,uint)
INSTANTIATE(uint,uint)
INSTANTIATE(uchar,uint)

}
