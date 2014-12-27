#include "math.hpp"

namespace opencl
{
    bool operator ==(cfloat a, cfloat b) { return (a.s[0] == b.s[0]) && (a.s[1] == b.s[1]); }
    bool operator !=(cfloat a, cfloat b) { return !(a == b); }
    bool operator ==(cdouble a, cdouble b) { return (a.s[0] == b.s[0]) && (a.s[1] == b.s[1]); }
    bool operator !=(cdouble a, cdouble b) { return !(a == b); }

    cfloat operator +(cfloat a, cfloat b)
    {
        cfloat res = {{a.s[0] + b.s[0], a.s[1] + b.s[1]}};
        return res;
    }

    cdouble operator +(cdouble a, cdouble b)
    {
        cdouble res = {{a.s[0] + b.s[0], a.s[1] + b.s[1]}};
        return res;
    }
}
