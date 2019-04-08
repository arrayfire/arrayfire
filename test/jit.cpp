/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define GTEST_LINKED_AS_SHARED_LIBRARY 1
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>

#include <numeric>

using af::array;
using af::constant;
using af::eval;
using af::freeHost;
using af::gforSet;
using af::randn;
using af::randu;
using af::seq;
using std::vector;
using std::iota;

TEST(JIT, CPP_JIT_HASH) {
    const int num     = 20;
    const float valA  = 3;
    const float valB  = 5;
    const float valC  = 2;
    const float valD  = valA + valB;
    const float valE  = valA + valC;
    const float valF1 = valD * valE - valE;
    const float valF2 = valD * valE - valD;

    array a = constant(valA, num);
    array b = constant(valB, num);
    array c = constant(valC, num);
    eval(a);
    eval(b);
    eval(c);

    // Creating a kernel
    {
        array d    = a + b;
        array e    = a + c;
        array f1   = d * e - e;
        float* hF1 = f1.host<float>();

        for (int i = 0; i < num; i++) { ASSERT_EQ(hF1[i], valF1); }

        freeHost(hF1);
    }

    // Making sure a different kernel is generated
    {
        array d    = a + b;
        array e    = a + c;
        array f2   = d * e - d;
        float* hF2 = f2.host<float>();

        for (int i = 0; i < num; i++) { ASSERT_EQ(hF2[i], valF2); }

        freeHost(hF2);
    }
}

TEST(JIT, CPP_JIT_Reset_Binary) {
    array a = constant(2, 5, 5);
    array b = constant(1, 5, 5);
    array c = a + b;
    array d = a - b;
    array e = c * d;
    e.eval();
    array f = c - d;
    f.eval();
    array g = d - c;
    g.eval();

    vector<float> hf(f.elements());
    vector<float> hg(g.elements());
    f.host(&hf[0]);
    g.host(&hg[0]);

    for (int i = 0; i < (int)f.elements(); i++) { ASSERT_EQ(hf[i], -hg[i]); }
}

TEST(JIT, CPP_JIT_Reset_Unary) {
    array a = constant(2, 5, 5);
    array b = constant(1, 5, 5);
    array c = sin(a);
    array d = cos(b);
    array e = c * d;
    e.eval();
    array f = c - d;
    f.eval();
    array g = d - c;
    g.eval();

    vector<float> hf(f.elements());
    vector<float> hg(g.elements());
    f.host(&hf[0]);
    g.host(&hg[0]);

    for (int i = 0; i < (int)f.elements(); i++) { ASSERT_EQ(hf[i], -hg[i]); }
}

TEST(JIT, CPP_Multi_linear) {
    const int num = 1 << 16;
    array a       = randu(num, s32);
    array b       = randu(num, s32);
    array x       = a + b;
    array y       = a - b;
    eval(x, y);

    vector<int> ha(num);
    vector<int> hb(num);
    vector<int> hx(num);
    vector<int> hy(num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ((ha[i] + hb[i]), hx[i]);
        ASSERT_EQ((ha[i] - hb[i]), hy[i]);
    }
}

TEST(JIT, CPP_strided) {
    const int num = 1024;
    gforSet(true);
    array a = randu(num, 1, s32);
    array b = randu(1, num, s32);
    array x = a + b;
    array y = a - b;
    eval(x);
    eval(y);
    gforSet(false);

    vector<int> ha(num);
    vector<int> hb(num);
    vector<int> hx(num * num);
    vector<int> hy(num * num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int j = 0; j < num; j++) {
        for (int i = 0; i < num; i++) {
            ASSERT_EQ((ha[i] + hb[j]), hx[j * num + i]);
            ASSERT_EQ((ha[i] - hb[j]), hy[j * num + i]);
        }
    }
}

TEST(JIT, CPP_Multi_strided) {
    const int num = 1024;
    gforSet(true);
    array a = randu(num, 1, s32);
    array b = randu(1, num, s32);
    array x = a + b;
    array y = a - b;
    eval(x, y);
    gforSet(false);

    vector<int> ha(num);
    vector<int> hb(num);
    vector<int> hx(num * num);
    vector<int> hy(num * num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int j = 0; j < num; j++) {
        for (int i = 0; i < num; i++) {
            ASSERT_EQ((ha[i] + hb[j]), hx[j * num + i]);
            ASSERT_EQ((ha[i] - hb[j]), hy[j * num + i]);
        }
    }
}

TEST(JIT, CPP_Multi_pre_eval) {
    const int num = 1 << 16;
    array a       = randu(num, s32);
    array b       = randu(num, s32);
    array x       = a + b;
    array y       = a - b;

    eval(x);

    // Should evaluate only y
    eval(x, y);

    // Should not evaluate anything
    // Should not error out
    eval(x, y);

    vector<int> ha(num);
    vector<int> hb(num);
    vector<int> hx(num);
    vector<int> hy(num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    x.host(&hx[0]);
    y.host(&hy[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ((ha[i] + hb[i]), hx[i]);
        ASSERT_EQ((ha[i] - hb[i]), hy[i]);
    }
}

TEST(JIT, CPP_common_node) {
    array r = seq(-3, 3, 0.5);

    int n = r.dims(0);

    array x = tile(r, 1, r.dims(0));
    array y = tile(r.T(), r.dims(0), 1);
    x.eval();
    y.eval();

    vector<float> hx(x.elements());
    vector<float> hy(y.elements());
    vector<float> hr(r.elements());

    x.host(&hx[0]);
    y.host(&hy[0]);
    r.host(&hr[0]);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(hx[j * n + i], hr[i]);
            ASSERT_EQ(hy[j * n + i], hr[j]);
        }
    }
}

TEST(JIT, ISSUE_1646) {
    array test1 = randn(10, 10);
    array test2 = randn(10);
    array test3 = randn(10);

    for (int i = 0; i < 1000; i++) {
        test3 += sum(test1, 1);
        test2 += test3;
    }
    eval(test2);
    eval(test3);
}

TEST(JIT, NonLinearLargeY) {
    const int d0 = 2;
    // This needs to be > 2 * (1 << 20) to properly check this.
    const int d1 = 3 * (1 << 20);
    array a      = randn(d0);
    array b      = randn(1, d1);

    // tile is jit-ted for both the operations
    array c = tile(a, 1, d1) + tile(b, d0, 1);
    eval(c);

    vector<float> ha(d0);
    vector<float> hb(d1);
    vector<float> hc(d0 * d1);

    a.host(ha.data());
    b.host(hb.data());
    c.host(hc.data());

    for (int j = 0; j < d1; j++) {
        for (int i = 0; i < d0; i++) {
            ASSERT_EQ(hc[i + j * d0], ha[i] + hb[j])
                << " at " << i << " , " << j;
        }
    }
}

TEST(JIT, NonLinearLargeX) {
    af_array r, c, s;
    dim_t rdims[] = {1024000, 1, 3};
    dim_t cdims[] = {1, 1, 3};
    dim_t sdims[] = {1, 1, 1};
    dim_t ndims   = 3;

    ASSERT_SUCCESS(af_randu(&r, ndims, rdims, f32));
    ASSERT_SUCCESS(af_constant(&c, 1, ndims, cdims, f32));
    ASSERT_SUCCESS(af_eval(c));
    ASSERT_SUCCESS(af_sub(&s, r, c, true));
    ASSERT_SUCCESS(af_eval(s));

    dim_t relem = 1;
    dim_t celem = 1;
    dim_t selem = 1;
    for (int i = 0; i < ndims; i++) {
        relem *= rdims[i];
        celem *= cdims[i];
        sdims[i] = std::max(rdims[i], cdims[i]);
        selem *= sdims[i];
    }

    vector<float> hr(relem);
    vector<float> hc(celem);
    vector<float> hs(selem);

    ASSERT_SUCCESS(af_get_data_ptr(hr.data(), r));
    ASSERT_SUCCESS(af_get_data_ptr(hc.data(), c));
    ASSERT_SUCCESS(af_get_data_ptr(hs.data(), s));

    for (int k = 0; k < sdims[2]; k++) {
        for (int j = 0; j < sdims[1]; j++) {
            for (int i = 0; i < sdims[0]; i++) {
                int sidx = i + j * sdims[0] + k * (sdims[0] * sdims[1]);

                int ridx = (i % rdims[0]) + (j % rdims[1]) * rdims[0] +
                           (k % rdims[2]) * rdims[0] * rdims[1];

                int cidx = (i % cdims[0]) + (j % cdims[1]) * cdims[0] +
                           (k % cdims[2]) * cdims[0] * cdims[1];

                ASSERT_EQ(hs[sidx], hr[ridx] - hc[cidx])
                    << " at " << i << "," << k;
            }
        }
    }

    ASSERT_SUCCESS(af_release_array(r));
    ASSERT_SUCCESS(af_release_array(c));
    ASSERT_SUCCESS(af_release_array(s));
}

TEST(JIT, ISSUE_1894) {
    array a = randu(1);
    array b = tile(a, 2 * (1 << 20));
    eval(b);
    float ha = -100;
    vector<float> hb(b.elements(), -200);

    a.host(&ha);
    b.host(hb.data());

    for (size_t i = 0; i < hb.size(); i++) { ASSERT_EQ(ha, hb[i]); }
}

TEST(JIT, LinearLarge) {
    // Needs to be larger than 65535 * 256 (or 1 << 24)
    float v1 = std::rand() % 100;
    float v2 = std::rand() % 100;

    array a = constant(v1, 1 << 25);
    array b = constant(v2, 1 << 25);
    array c = (a + b) * (a - b);
    eval(c);

    float v3 = (v1 + v2) * (v1 - v2);

    vector<float> hc(c.elements());
    c.host(hc.data());

    for (size_t i = 0; i < hc.size(); i++) { ASSERT_EQ(hc[i], v3); }
}

TEST(JIT, NonLinearBuffers1) {
    array a  = randu(5, 5);
    array a0 = a;
    for (int i = 0; i < 1000; i++) {
        array b = randu(1, 5);
        a += tile(b, 5);
    }
    a.eval();
}

TEST(JIT, NonLinearBuffers2) {
    array a = randu(100, 310);
    array b = randu(10, 10);
    for (int i = 0; i < 300; i++) {
        b += a(seq(10), seq(i, i + 9)) * randu(10, 10);
    }
    b.eval();
}

TEST(JIT, TransposeBuffers) {
    const int num = 10;
    array a       = randu(1, num);
    array b       = randu(1, num);
    array c       = a + b;
    array d       = a.T() + b.T();

    vector<float> ha(a.elements());
    a.host(ha.data());

    vector<float> hb(b.elements());
    b.host(hb.data());

    vector<float> hc(c.elements());
    c.host(hc.data());

    vector<float> hd(d.elements());
    d.host(hd.data());

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(ha[i] + hb[i], hc[i]);
        ASSERT_FLOAT_EQ(hc[i], hd[i]);
    }
}

using af::dim4;

struct tile_params {
    dim4 in_dim;
    dim4 tile;
    dim4 out_dim;
    tile_params(dim4 in, dim4 t, dim4 out)
        : in_dim(in), tile(t), out_dim(out) {}
};

std::ostream& operator<<(std::ostream& os, const tile_params& tp) {
    os << "in_dim: " << tp.in_dim << "; tile parameters: " << tp.tile
       << "; out_dim " << tp.out_dim << ";";
    return os;
}

class JIT : public ::testing::TestWithParam<tile_params> {
   protected:
    void SetUp() {
        tile_params params = GetParam();
        vector<float> vals(params.in_dim.elements());
        iota(vals.begin(), vals.end(), 0);
        in = array(params.in_dim, &vals.front());

        // clang-format off
        gold.resize(params.out_dim.elements());
        dim_t tile_dim[4] = {params.tile[0], params.tile[1], params.tile[2],
                             params.tile[3]};

        dim_t istride[4] = {1,
                            params.in_dim[0],
                            params.in_dim[0] * params.in_dim[1],
                            params.in_dim[0] * params.in_dim[1] * params.in_dim[2]};
        dim_t ostride[4] = {1,
                            params.out_dim[0],
                            params.out_dim[0] * params.out_dim[1],
                            params.out_dim[0] * params.out_dim[1] * params.out_dim[2]};

        for (int i = 0; i < 4; i++) {
            if (tile_dim[i] != 1) { istride[i] = 0; }
        }

        for (int l = 0; l < params.out_dim[3]; l++) {
            for (int k = 0; k < params.out_dim[2]; k++) {
                for (int j = 0; j < params.out_dim[1]; j++) {
                    for (int i = 0; i < params.out_dim[0]; i++) {
                        gold[l * ostride[3] +
                            k * ostride[2] +
                            j * ostride[1] +
                            i * ostride[0]] = vals[l * istride[3] +
                                                    k * istride[2] +
                                                    j * istride[1] +
                                                    i * istride[0]];
                    }
                }
            }
        }
        // clang-format on
    }

   public:
    array in;
    vector<float> gold;
};

void replace_all(std::string& str, const std::string& oldStr,
                 const std::string& newStr) {
    std::string::size_type pos = 0u;
    while ((pos = str.find(oldStr, pos)) != std::string::npos) {
        str.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }
}

std::string concat_dim4(dim4 d) {
    std::stringstream ss;
    ss << d;
    std::string s = ss.str();
    replace_all(s, " ", "_");
    return s;
}
std::string tile_info(const ::testing::TestParamInfo<JIT::ParamType> info) {
    std::stringstream ss;
    ss << "in_" << concat_dim4(info.param.in_dim) << "_tile_"
       << concat_dim4(info.param.tile);
    return ss.str();
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
                        JitTile, JIT,
                                                   //  input_dim            tile_dim             output_dim
                        ::testing::Values(
                                          tile_params( dim4(10),            dim4(1, 10),         dim4(10, 10)),
                                          tile_params( dim4(10),            dim4(1, 1, 10),      dim4(10, 1, 10)),
                                          tile_params( dim4(10),            dim4(1, 1, 1, 10),   dim4(10, 1, 1, 10)),
                                          tile_params( dim4(1, 10),         dim4(10),            dim4(10, 10)),
                                          tile_params( dim4(1, 10),         dim4(1, 1, 10),      dim4(1, 10, 10)),
                                          tile_params( dim4(1, 10),         dim4(1, 1, 1, 10),   dim4(1, 10, 1, 10)),

                                          tile_params( dim4(10, 10),        dim4(1, 1, 10),      dim4(10, 10, 10)),
                                          tile_params( dim4(10, 10),        dim4(1, 1, 1, 10),   dim4(10, 10, 1, 10)),

                                          tile_params( dim4(1, 1, 10),      dim4(10),            dim4(10, 1, 10)),
                                          tile_params( dim4(1, 1, 10),      dim4(1, 10),         dim4(1, 10, 10)),
                                          tile_params( dim4(1, 1, 10),      dim4(1, 1, 1, 10),   dim4(1, 1, 10, 10)),

                                          tile_params( dim4(1, 10, 10),     dim4(10),            dim4(10, 10, 10)),
                                          tile_params( dim4(10, 1, 10),     dim4(1, 10),         dim4(10, 10, 10)),
                                          tile_params( dim4(10, 1, 10),     dim4(1, 1, 1, 10),   dim4(10, 1, 10, 10)),
                                          tile_params( dim4(1, 10, 10),     dim4(1, 1, 1, 10),   dim4(1, 10, 10, 10)),
                                          tile_params( dim4(10, 10, 10),    dim4(1, 1, 1, 10),   dim4(10, 10, 10, 10)),

                                          tile_params( dim4(1, 1, 1, 10),   dim4(10),            dim4(10, 1, 1, 10)),
                                          tile_params( dim4(1, 10, 1, 10),  dim4(10),            dim4(10, 10, 1, 10)),
                                          tile_params( dim4(1, 1, 10, 10),  dim4(10),            dim4(10, 1, 10, 10)),
                                          tile_params( dim4(1, 10, 10, 10), dim4(10),            dim4(10, 10, 10, 10)),

                                          tile_params( dim4(1, 1, 1, 10),   dim4(1, 10),         dim4(1, 10, 1, 10)),
                                          tile_params( dim4(10, 1, 1, 10),  dim4(1, 10),         dim4(10, 10, 1, 10)),
                                          tile_params( dim4(1, 1, 10, 10),  dim4(1, 10),         dim4(1, 10, 10, 10)),

                                          tile_params( dim4(1, 1, 1, 10),   dim4(1, 1, 10),      dim4(1, 1, 10, 10)),
                                          tile_params( dim4(10, 1, 1, 10),  dim4(1, 1, 10),      dim4(10, 1, 10, 10)),
                                          tile_params( dim4(1, 10, 1, 10),  dim4(1, 1, 10),      dim4(1, 10, 10, 10)),
                                          tile_params( dim4(10, 10, 1, 10), dim4(1, 1, 10),      dim4(10, 10, 10, 10))
                                          ),
                        tile_info
                        );
// clang-format on

TEST_P(JIT, Tile) {
    tile_params params = GetParam();
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;
    size_t alloc_bytes2, alloc_buffers2;
    size_t lock_bytes2, lock_buffers2;
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    array out = tile(in, params.tile);
    af::deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2,
                      &lock_buffers2);

    // Make sure that the dimensions we are testing here are JIT nodes
    // by checking that no new buffers are created.
    ASSERT_EQ(alloc_bytes, alloc_bytes2)
        << "Tile operation created a buffer therefore not a JIT node";
    ASSERT_EQ(alloc_buffers, alloc_buffers2)
        << "Tile operation created a buffer therefore not a JIT node";
    ASSERT_EQ(lock_bytes, lock_bytes2)
        << "Tile operation created a buffer therefore not a JIT node";
    ASSERT_EQ(alloc_buffers, alloc_buffers2)
        << "Tile operation created a buffer therefore not a JIT node";

    ASSERT_VEC_ARRAY_EQ(gold, params.out_dim, out);
}
