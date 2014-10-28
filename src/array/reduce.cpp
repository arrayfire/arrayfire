#include <af/array.h>
#include <af/reduce.h>
#include "error.hpp"

namespace af
{
    array sum(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_sum(&out, in.get(), dim));
        return array(out);
    }

    array min(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_min(&out, in.get(), dim));
        return array(out);
    }

    array max(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_max(&out, in.get(), dim));
        return array(out);
    }

    array alltrue(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_alltrue(&out, in.get(), dim));
        return array(out);
    }

    array anytrue(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_anytrue(&out, in.get(), dim));
        return array(out);
    }

    array count(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_count(&out, in.get(), dim));
        return array(out);
    }

}
