#include <af/array.h>
#include <copy.hpp>
#include <print.hpp>
#include <ArrayInfo.hpp>
#include <helper.h>
#include <backend.h>

using namespace detail;
af_err af_print(af_array arr)
{
    af_err ret = AF_ERR_RUNTIME;
    try {
        ArrayInfo info = getInfo(arr);
        switch(info.getType())
        {
            case f32:   print(getArray<float>(arr));    break;
            case c32:   print(getArray<cfloat>(arr));   break;
            case f64:   print(getArray<double>(arr));   break;
            case c64:   print(getArray<cdouble>(arr));  break;
            case b8:    print(getArray<char>(arr));     break;
            case s32:   print(getArray<int>(arr));      break;
            case u32:   print(getArray<unsigned>(arr)); break;
            case u8:    print(getArray<uchar>(arr));    break;
            case s8:    print(getArray<char>(arr));     break;
            default:    ret = AF_ERR_NOT_SUPPORTED;
        }
    }
    CATCHALL

    return ret;
}
