
#include <iostream>
#include <print.hpp>
#include <Array.hpp>
#include <iostream>

namespace cpu
{
    using std::ostream;
    using std::endl;

    template<typename T>
    void
    printer(ostream &out, const T* ptr, const Array<T> &info, unsigned dim)
    {

        dim_type stride =   info.strides()[dim];
        dim_type d      =   info.dims()[dim];

        if(dim == 0) {
            for(dim_type i = 0, j = 0; i < d; i++, j+=stride) {
                out << ptr[j] << "\t";
            }
            out << endl;
        }
        else {
            for(int i = 0; i < d; i++) {
                printer(out, ptr, info, dim - 1);
                ptr += stride;
            }
        }

    }

    template<typename T>
    void
    operator <<(ostream &out, const Array<T> &arr)
    {
        out << "TRANSPOSED\n";
        out << "Dim:" << arr.dims();
        out << "Offset: " << arr.offsets();
        out << "Stride: " << arr.strides();
        printer(out, arr.get(), arr, arr.ndims() - 1);
    }

    template<typename T>
    void
    print(const af_array &arr) {
        const Array<T> &impl = getArray<T>(arr);
        std::cout << impl;
    }

    template void print<float>                   (const af_array &arr);
    template void print<cfloat>                  (const af_array &arr);
    template void print<double>                  (const af_array &arr);
    template void print<cdouble>                 (const af_array &arr);
    template void print<char>                    (const af_array &arr);
    template void print<int>                     (const af_array &arr);
    template void print<unsigned>                (const af_array &arr);
    template void print<uchar>                   (const af_array &arr);
}
