#include <ArrayInfo.hpp>
#include <numeric>
#include <algorithm>
#include <functional>

using af::dim4;
using std::partial_sum;
using std::accumulate;
using std::transform;

dim_type
calcGlobalOffset(const ArrayInfo &info, const ArrayInfo &parentInfo)
{

    size_t ndims = info.ndims();
    const dim_type *offsetPtr = info.offsets().get();
    const dim_type *dimPtr = parentInfo.dims().get();
    dim4 out(offsetPtr[0],0,0,0);

    transform(  offsetPtr + 1, offsetPtr + ndims,
                dimPtr,
                out.get() + 1,
                std::multiplies<dim_type>());

    return abs(accumulate(out.get(), out.get() + ndims, 0));
}


const ArrayInfo&
getInfo(af_array arr)
{
    const ArrayInfo *info = static_cast<ArrayInfo*>(reinterpret_cast<void *>(arr));
    return *info;
}

dim_type
af_get_elements(af_array arr)
{
    return getInfo(arr).elements();
}

af_dtype af_get_type(af_array arr)
{
    return getInfo(arr).getType();
}

dim4
calcBaseStride(const dim4 &parentDim)
{
    dim4 out(1, 1, 1, 1);
    const dim_type *parentPtr = parentDim.get();
    partial_sum(parentPtr, parentPtr + parentDim.ndims(), out.get() + 1);
    return out;
}

