#include <af/image.h>
#include "error.hpp"

namespace af
{

array medfilt(const array& in, dim_type wind_length, dim_type wind_width, af_pad_type edge_pad)
{
    af_array out = 0;
    AF_THROW(af_medfilt(&out, in.get(), wind_length, wind_width, edge_pad));
    return array(out);
}

}
