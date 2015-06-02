/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <arrayfire.h>

union Data
{
    unsigned dim;
    char bytes[4];
};

unsigned char reverse_char(unsigned char b)
{
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}

// http://stackoverflow.com/a/9144870/2192361
unsigned reverse(unsigned x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

template<class ty>
void read_idx(std::vector<dim_t> &dims, std::vector<ty> &data, const char *name)
{
    std::ifstream f(name, std::ios::in | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Unable to open file");

    Data d;
    f.read(d.bytes, sizeof(d.bytes));

    if (d.bytes[2] != 8) {
        throw std::runtime_error("Unsupported data type");
    }

    unsigned numdims = d.bytes[3];
    unsigned elemsize = 1;

    // Read the dimensions
    size_t elem = 1;
    dims = std::vector<dim_t>(numdims);
    for (unsigned i = 0; i < numdims; i++) {
        f.read(d.bytes, sizeof(d.bytes));

        // Big endian to little endian
        for (int j = 0; j < 4; j++) d.bytes[j] = reverse_char(d.bytes[j]);
        unsigned dim = reverse(d.dim);

        elem *= dim;
        dims[i] = (dim_t)dim;
    }

    // Read the data
    std::vector<char> cdata(elem);
    f.read(&cdata[0], elem * elemsize);
    std::vector<unsigned char> ucdata(cdata.begin(), cdata.end());

    data = std::vector<ty>(ucdata.begin(), ucdata.end());

    f.close();
    return;
}
