#include <Array.hpp>
#include <af/dim4.hpp>
#include <cassert>
#include <vector>

namespace cuda
{
    //using af::toOffset;
    //using af::toStride;
    //using af::toDims;
    using std::vector;

template<typename T>
void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq* index)
{
    if(dest)    {   assert("NOT IMPLEMENTED" && 1 != 1); }
    else {
        vector<af_seq> index_(index, index+ndims);
        const Array<T> &parent = getArray<T>(src);
        Array<T>* dst =  createView(    parent,
                                            af::toDims(index_, parent.dims()),
                                            af::toOffset(index_),
                                            af::toStride(index_, parent.dims()) );
        dest = getHandle(*dst);
    }
}
template void indexArray<float>         (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<cfloat>        (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<double>        (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<cdouble>       (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<unsigned>      (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<int>           (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<uchar>         (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
template void indexArray<char>          (af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);
}
