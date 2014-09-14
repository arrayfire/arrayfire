#include <iostream>
#include <af/array.h>
#include <copy.hpp>
#include <print.hpp>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <backend.hpp>

using namespace detail;
using std::ostream;
using std::cout;
using std::endl;

//uchar to number converters
template<typename T>
struct ToNum
{
    inline T operator()(T val) { return val; }
};

template<>
struct ToNum<unsigned char>
{
    inline int operator()(unsigned char val) { return static_cast<int>(val); }
};

template<typename T>
static void printer(ostream &out, const T* ptr, const ArrayInfo &info, unsigned dim)
{

    dim_type stride =   info.strides()[dim];
    dim_type d      =   info.dims()[dim];
    ToNum<T> toNum;

    if(dim == 0) {
        for(dim_type i = 0, j = 0; i < d; i++, j+=stride) {
            out << toNum(ptr[j]) << "\t";
        }
        out << endl;
    }
    else {
        for(dim_type i = 0; i < d; i++) {
            printer(out, ptr, info, dim - 1);
            ptr += stride;
        }
        out << endl;
    }
}

template<typename T>
static void print(af_array arr)
{
    const ArrayInfo info = getInfo(arr);
    T *data = new T[info.elements()];
    //FIXME: Use alternative function to avoid copies if possible
    af_get_data_ptr(data, arr);

    std::cout << "TRANSPOSED\n";
    std::cout << "Dim:" << info.dims();
    std::cout << "Offset: " << info.offsets();
    std::cout << "Stride: " << info.strides();

    printer(std::cout, data, info, info.ndims() - 1);

    delete[] data;
}

af_err af_print(af_array arr)
{
    af_err ret = AF_ERR_RUNTIME;
    try {
        ArrayInfo info = getInfo(arr);
        switch(info.getType())
        {
            case f32:   print<float>(arr);    break;
            case c32:   print<cfloat>(arr);   break;
            case f64:   print<double>(arr);   break;
            case c64:   print<cdouble>(arr);  break;
            case b8:    print<char>(arr);     break;
            case s32:   print<int>(arr);      break;
            case u32:   print<unsigned>(arr); break;
            case u8:    print<uchar>(arr);    break;
            case s8:    print<char>(arr);     break;
            default:    ret = AF_ERR_NOT_SUPPORTED;
        }
    }
    CATCHALL

    return ret;
}
