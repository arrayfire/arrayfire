#pragma once
#include <cl.hpp>
#include <kernel/KParam.hpp>
typedef struct
{
    cl::Buffer data;
    KParam info;
} Param;
