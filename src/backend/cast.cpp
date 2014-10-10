#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <implicit.hpp>

#include <cast.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>

af_err af_cast(af_array *out, const af_array in, const af_dtype type)
{
    try {
        af_array res = cast(in, type);
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
