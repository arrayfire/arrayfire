#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <elwise.hpp>
#include <common_helper.h>
#include <backend_helper.h>

af_err af_add(af_array *result, const af_array lhs, const af_array rhs)
{
    af_err ret = AF_SUCCESS;
    try {
        af_dtype lhs_t = af_get_type(lhs);
        af_dtype rhs_t = af_get_type(rhs);

        getFunction(lhs_t , rhs_t)(result, lhs, rhs);
    }
    CATCHALL

    return ret;
}
