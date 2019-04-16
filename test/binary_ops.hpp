#include <af/defines.h>
#include <algorithm>
#include <complex>
#include <numeric>

template<typename T>
static inline T min(T lhs, T rhs) {
    return std::min(lhs, rhs);
}
std::complex<float> min(std::complex<float> lhs, std::complex<float> rhs);
std::complex<double> min(std::complex<double> lhs, std::complex<double> rhs);

template<typename T>
static inline T max(T lhs, T rhs) {
    return std::max(lhs, rhs);
}
std::complex<float> max(std::complex<float> lhs, std::complex<float> rhs);
std::complex<double> max(std::complex<double> lhs, std::complex<double> rhs);

template<typename T, af_binary_op op>
struct Binary {
    T init() { return (T)(0); }

    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

template<typename T>
struct Binary<T, AF_BINARY_ADD> {
    T init() { return (T)(0); }

    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

template<typename T>
struct Binary<T, AF_BINARY_MUL> {
    T init() { return (T)(1); }

    T operator()(T lhs, T rhs) { return lhs * rhs; }
};

template<typename T>
struct Binary<T, AF_BINARY_MIN> {
    T init() { return std::numeric_limits<T>::max(); }

    T operator()(T lhs, T rhs) { return min(lhs, rhs); }
};

template<typename T>
struct Binary<T, AF_BINARY_MAX> {
    T init() { return std::numeric_limits<T>::min(); }

    T operator()(T lhs, T rhs) { return max(lhs, rhs); }
};

#define SPECIALIZE_COMPLEX_MIN(T, Tr)                            \
    template<>                                                   \
    struct Binary<T, AF_BINARY_MIN> {                            \
        T init() { return (T)(std::numeric_limits<Tr>::max()); } \
                                                                 \
        T operator()(T lhs, T rhs) { return min(lhs, rhs); }     \
    };

SPECIALIZE_COMPLEX_MIN(std::complex<float>, float)
SPECIALIZE_COMPLEX_MIN(std::complex<double>, double)
#undef SPECIALIZE_COMPLEX_MIN

#define SPECIALIZE_COMPLEX_MAX(T, Tr)                        \
    template<>                                               \
    struct Binary<T, AF_BINARY_MAX> {                        \
        T init() { return (T)((Tr)(0)); }                    \
                                                             \
        T operator()(T lhs, T rhs) { return max(lhs, rhs); } \
    };

SPECIALIZE_COMPLEX_MAX(std::complex<float>, float)
SPECIALIZE_COMPLEX_MAX(std::complex<double>, double)
#undef SPECIALIZE_COMPLEX_MAX

#define SPECIALIZE_FLOATING_MAX(T, Tr)                            \
    template<>                                                    \
    struct Binary<T, AF_BINARY_MAX> {                             \
        T init() { return (T)(-std::numeric_limits<Tr>::max()); } \
                                                                  \
        T operator()(T lhs, T rhs) { return max(lhs, rhs); }      \
    };

SPECIALIZE_FLOATING_MAX(float, float)
SPECIALIZE_FLOATING_MAX(double, double)
#undef SPECIALIZE_FLOATING_MAX
