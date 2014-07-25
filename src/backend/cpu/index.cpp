#include <index.hpp>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <cassert>

namespace cpu
{

template<typename T>
void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index)
{
    using af::toOffset;
    using af::toDims;
    using af::toStride;

    if(dest)    {   assert("NOT IMPLEMENTED" && 1 != 1); }
    else {
        const Array<T> &parent = getArray<T>(src);
        vector<af_seq> index_(index, index+ndims);
        Array<T>* dst =  createView(    parent,
                                            toDims(index_, parent.dims()),
                                            toOffset(index_),
                                            toStride(index_, parent.dims()) );
        dest = getHandle(*dst);
    }
}

template void indexArray<float>         (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<cfloat>        (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<double>        (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<cdouble>       (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<unsigned>      (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<int>           (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<unsigned char> (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<char>          (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);

}
