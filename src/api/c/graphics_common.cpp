/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <graphics_common.hpp>
#include <err_common.hpp>

GLenum glErrorSkip(const char *msg, const char* file, int line)
{
#ifndef NDEBUG
    GLenum x = glGetError();
    if (x != GL_NO_ERROR) {
        printf("GL Error Skipped at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
    }
    return x;
#else
    return 0;
#endif
}

GLenum glErrorCheck(const char *msg, const char* file, int line)
{
// Skipped in release mode
#ifndef NDEBUG
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        printf("GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AF_ERROR("Error in Graphics", AF_ERR_GL_ERROR);
    }
    return x;
#else
    return 0;
#endif
}

GLenum glForceErrorCheck(const char *msg, const char* file, int line)
{
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        printf("GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AF_ERROR("Error in Graphics", AF_ERR_GL_ERROR);
    }
    return x;
}

#endif
