#pragma once
#include <string>
#include <af/defines.h>

const char *getName(af_dtype type);

//uchar to number converters
template<typename T>
struct ToNum
{
    inline T operator()(T val) { return val; }
};

template<>
struct ToNum<unsigned char>
{
    inline int operator()(unsigned char val) { return static_cast<int>(val); }
};
