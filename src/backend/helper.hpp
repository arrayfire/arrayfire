#pragma once

unsigned size_of(af_dtype type);

#define CATCHALL                \
    catch(...) {                \
        return AF_ERR_INTERNAL; \
    }

//FIXME: This doesn't belong here. maybe traits.hpp?
#include <backend.hpp>
#include <handle.hpp>
template<typename T> struct is_complex                  { static const bool value = false;  };
template<> struct           is_complex<detail::cfloat>  { static const bool value = true;   };
template<> struct           is_complex<detail::cdouble> { static const bool value = true;   };

//uchar to number converters
template<typename T>
struct ToNum{
    inline T operator()(T val) { return val; }
};

template<>
struct ToNum<unsigned char>{
    inline int operator()(unsigned char val) { return static_cast<int>(val); }
};
