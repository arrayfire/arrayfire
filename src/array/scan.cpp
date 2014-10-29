#include <af/array.h>
#include <af/data.h>
#include "error.hpp"

namespace af
{
    array accum(const array& in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_accum(&out, in.get(), dim));
        return array(out);
    }
}
