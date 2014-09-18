#pragma once

namespace cuda
{
namespace kernel
{

    static const uint THREADS_PER_BLOCK = 256;
    static const uint THREADS_X = 32;
    static const uint THREADS_Y = THREADS_PER_BLOCK / THREADS_X;
    static const uint REPEAT    = 32;
}
}
