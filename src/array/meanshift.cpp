#include <af/image.h>
#include "error.hpp"

namespace af
{

array meanshift(const array& in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color)
{
    af_array out = 0;
    AF_THROW(af_meanshift(&out, in.get(), spatial_sigma, chromatic_sigma, iter, is_color));
    return array(out);
}

}
