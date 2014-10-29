#include <af/image.h>
#include "error.hpp"

namespace af
{

array histogram(const array &in, const unsigned nbins, const double minval, const double maxval)
{
    af_array out = 0;
    AF_THROW(af_histogram(&out, in.get(), nbins, minval, maxval));
    return array(out);
}

}
