
#include <common/half.hpp>
#include <common/util.hpp>

namespace arrayfire {
namespace common {
std::ostream &operator<<(std::ostream &os, const half &val) {
    os << float(val);
    return os;
}

template<>
std::string toString(const half val) {
    return common::toString(static_cast<float>(val));
}
}  // namespace common
}  // namespace arrayfire
