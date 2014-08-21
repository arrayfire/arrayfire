#include <ArrayInfo.hpp>
#include <numeric>
#include <algorithm>
#include <functional>

using af::dim4;
using std::partial_sum;
using std::accumulate;
using std::transform;

dim_type
calcGlobalOffset(const af::dim4 &strides, const af::dim4 &offsets)
{
    dim_type offset = 0;
    for (int i = 0; i < 4; i++) offset += offsets[i] * strides[i];
    return offset;
}


const ArrayInfo&
getInfo(af_array arr)
{
    const ArrayInfo *info = static_cast<ArrayInfo*>(reinterpret_cast<void *>(arr));
    return *info;
}

af_err
af_get_elements(dim_type *elems, const af_array arr)
{
    *elems =  getInfo(arr).elements();
    return AF_SUCCESS; //FIXME: Catch exceptions correctly
}

af_err af_get_type(af_dtype *type, const af_array arr)
{
    *type = getInfo(arr).getType();
    return AF_SUCCESS; //FIXME: Catch exceptions correctly
}

dim4 calcBaseStride(const dim4 &parentDim)
{
    dim4 out(1, 1, 1, 1);
    const dim_type *parentPtr = parentDim.get();
    partial_sum(parentPtr, parentPtr + parentDim.ndims(), out.get() + 1, std::multiplies<dim_type>());
    return out;
}

void ArrayInfo::moddims(const dim4 &newDims)
{
    dim_size   = newDims;
    strides()  = calcBaseStride(newDims);
}
