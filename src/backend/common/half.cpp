
#include <common/half.hpp>

namespace common {
std::ostream &operator<<(std::ostream &os, const half &val) {
    os << float(val);
    return os;
}
}  // namespace common
