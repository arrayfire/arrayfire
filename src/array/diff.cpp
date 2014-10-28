#include <af/array.h>
#include <af/data.h>
#include "error.hpp"

namespace af
{
    array diff1(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_diff1(&out, in.get(), dim));
        return array(out);
    }

    array diff2(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_diff2(&out, in.get(), dim));
        return array(out);
    }
}
