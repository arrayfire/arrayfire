#include <ArrayInfo.hpp>
#include <numeric>
#include <algorithm>
#include <functional>

using af::dim4;

dim_type
calcOffset(const af::dim4 &strides, const af::dim4 &offsets)
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

dim4 calcStrides(const dim4 &parentDim)
{
    dim4 out(1, 1, 1, 1);
    dim_type *out_dims = out.get();
    const dim_type *parent_dims =  parentDim.get();

    for (dim_type i=1; i < 4; i++) {
        out_dims[i] = out_dims[i - 1] * parent_dims[i-1];
    }

    return out;
}

void ArrayInfo::modStrides(const dim4 &newStrides)
{
    dim_strides = newStrides;
}

void ArrayInfo::modDims(const dim4 &newDims)
{
    dim_size   = newDims;
    modStrides(calcStrides(newDims));
}

bool ArrayInfo::isEmpty()
{
    return (elements() == 0);
}

bool ArrayInfo::isScalar()
{
    return (elements() == 1);
}

bool ArrayInfo::isRow()
{
    return (ndims() == 2 && dims()[0] == 1);
}

bool ArrayInfo::isColumn()
{
    return (ndims() == 1);
}

bool ArrayInfo::isVector()
{
    bool ret = true;
    for(unsigned i = 0; i < (ndims() - 1) && ret; i++) {
        ret = (dims()[i] == 1);
    }
    return ret;
}

bool ArrayInfo::isComplex()
{
    return ((type == c32) || (type == c64));
}

bool ArrayInfo::isReal()
{
    return !isComplex();
}

bool ArrayInfo::isDouble()
{
    return (type == f64 || type == c64);
}

bool ArrayInfo::isSingle()
{
    return (type == f32 || type == c32);
}

bool ArrayInfo::isRealFloating()
{
    return (type == f64 || type == f32);
}

bool ArrayInfo::isFloating()
{
    return (!isInteger());
}

bool ArrayInfo::isInteger()
{
    return (type == s32
         || type == u32
         || type == u8);
}
