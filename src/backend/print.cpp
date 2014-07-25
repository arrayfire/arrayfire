#include <af/array.h>
#include <copy.hpp>
#include <print.hpp>
#include <ArrayInfo.hpp>
#include <common_helper.h>
#include <backend_helper.h>

af_err
af_print(af_array arr)
{
    af_err ret = AF_ERR_RUNTIME;
    try {
        ArrayInfo info = getInfo(arr);
        switch(info.getType())
        {
            case f32:   print<float>(arr);          break;
            case c32:   print<cfloat>(arr);         break;
            case f64:   print<double>(arr);         break;
            case c64:   print<cdouble>(arr);        break;
            case b8:    print<char>(arr);           break;
            case s32:   print<int>(arr);            break;
            case u32:   print<unsigned>(arr);       break;
            case u8:    print<unsigned char>(arr);  break;
            case s8:    print<char>(arr);           break;
            default:    ret = AF_ERR_NOT_SUPPORTED;
        }
    }
    CATCHALL

    return ret;
}
