/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

int lIndex(const int j, const int i) { return j * SHRD_MEM_WIDTH + i; }

int gIndex(const int x, const int y, const int dim0, const int dim1,
           const int stride0, const int stride1) {
    return clamp(x, 0, dim0 - 1) * stride0 + clamp(y, 0, dim1 - 1) * stride1;
}

float quadratic(const float value) { return 1.0f / (1.0f + value); }

float computeGradientBasedUpdate(const float mct, const float C, const float S,
                                 const float N, const float W, const float E,
                                 const float SE, const float SW, const float NE,
                                 const float NW, const int FLUX_FN) {
    float delta = 0;

    float dx, dy, df, db, cx, cxd;

    // centralized derivatives
    dx = (E - W) * 0.5f;
    dy = (S - N) * 0.5f;

    // half-d's and conductance along first dimension
    df = E - C;
    db = C - W;

    if (FLUX_FN == 2) {
        cx  = exp((df * df + 0.25f * pow(dy + 0.5f * (SE - NE), 2)) * mct);
        cxd = exp((db * db + 0.25f * pow(dy + 0.5f * (SW - NW), 2)) * mct);
    } else {
        cx = quadratic((df * df + 0.25f * pow(dy + 0.5f * (SE - NE), 2)) * mct);
        cxd =
            quadratic((db * db + 0.25f * pow(dy + 0.5f * (SW - NW), 2)) * mct);
    }

    delta += (cx * df - cxd * db);

    // half-d's and conductance along second dimension
    df = S - C;
    db = C - N;

    if (FLUX_FN == 2) {
        cx  = exp((df * df + 0.25f * pow(dx + 0.5f * (SE - SW), 2)) * mct);
        cxd = exp((db * db + 0.25f * pow(dx + 0.5f * (NE - NW), 2)) * mct);
    } else {
        cx = quadratic((df * df + 0.25f * pow(dx + 0.5f * (SE - SW), 2)) * mct);
        cxd =
            quadratic((db * db + 0.25f * pow(dx + 0.5f * (NE - NW), 2)) * mct);
    }

    delta += (cx * df - cxd * db);

    return delta;
}

float computeCurvatureBasedUpdate(const float mct, const float C, const float S,
                                  const float N, const float W, const float E,
                                  const float SE, const float SW,
                                  const float NE, const float NW,
                                  const int FLUX_FN) {
    float delta     = 0;
    float prop_grad = 0;

    float df0, db0;
    float dx, dy, df, db, cx, cxd, gmf, gmb, gmsqf, gmsqb;

    // centralized derivatives
    dx = (E - W) * 0.5f;
    dy = (S - N) * 0.5f;

    // half-d's and conductance along first dimension
    df  = E - C;
    db  = C - W;
    df0 = df;
    db0 = db;

    gmsqf = (df * df + 0.25f * pow(dy + 0.5f * (SE - NE), 2));
    gmsqb = (db * db + 0.25f * pow(dy + 0.5f * (SW - NW), 2));

    gmf = sqrt(1.0e-10f + gmsqf);
    gmb = sqrt(1.0e-10f + gmsqb);

    cx  = exp(gmsqf * mct);
    cxd = exp(gmsqb * mct);

    delta += ((df / gmf) * cx - (db / gmb) * cxd);

    // half-d's and conductance along second dimension
    df = S - C;
    db = C - N;

    gmsqf = (df * df + 0.25f * pow(dx + 0.5f * (SE - SW), 2));
    gmsqb = (db * db + 0.25f * pow(dx + 0.5f * (NE - NW), 2));

    gmf = sqrt(1.0e-10 + gmsqf);
    gmb = sqrt(1.0e-10 + gmsqb);

    cx  = exp(gmsqf * mct);
    cxd = exp(gmsqb * mct);

    delta += ((df / gmf) * cx - (db / gmb) * cxd);

    if (delta > 0) {
        prop_grad += (pow(min(db0, 0.0f), 2.0f) + pow(max(df0, 0.0f), 2.0f));
        prop_grad += (pow(min(db, 0.0f), 2.0f) + pow(max(df, 0.0f), 2.0f));
    } else {
        prop_grad += (pow(max(db0, 0.0f), 2.0f) + pow(min(df0, 0.0f), 2.0f));
        prop_grad += (pow(max(db, 0.0f), 2.0f) + pow(min(df, 0.0f), 2.0f));
    }

    return sqrt(prop_grad) * delta;
}

kernel void diffUpdate(global T* inout, KParam info, const float dt,
                       const float mct, const int FLUX_FN, unsigned blkX,
                       unsigned blkY) {
    // Beware of the integer value of FLUX_FN

    local T localMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const unsigned b2 = get_group_id(0) / blkX;
    const unsigned b3 = get_group_id(1) / blkY;

    const int gx = get_local_size(0) * (get_group_id(0) - b2 * blkX) + lx;
    const int gy = get_local_size(1) * (get_group_id(1) - b3 * blkY) + ly;

    global T* img =
        inout + (b3 * info.strides[3] + b2 * info.strides[2]) + info.offset;

    for (int b = ly, gy2 = gy; b < SHRD_MEM_HEIGHT;
         b += get_local_size(1), gy2 += get_local_size(1)) {
        for (int a = lx, gx2 = gx; a < SHRD_MEM_WIDTH;
             a += get_local_size(0), gx2 += get_local_size(0)) {
            int idx = gIndex(gx2 - 1, gy2 - 1, info.dims[0], info.dims[1],
                             info.strides[0], info.strides[1]);
            localMem[b][a] = img[idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < info.dims[0] && gy < info.dims[1]) {
        int i       = lx + 1;
        int j       = ly + 1;
        float C     = localMem[j][i];
        float delta = 0;

#if IS_MCDE == 1
        delta = computeCurvatureBasedUpdate(
            mct, C, localMem[j][i + 1], localMem[j][i - 1], localMem[j - 1][i],
            localMem[j + 1][i], localMem[j + 1][i + 1], localMem[j - 1][i + 1],
            localMem[j + 1][i - 1], localMem[j - 1][i - 1], FLUX_FN);
#else
        delta = computeGradientBasedUpdate(
            mct, C, localMem[j][i + 1], localMem[j][i - 1], localMem[j - 1][i],
            localMem[j + 1][i], localMem[j + 1][i + 1], localMem[j - 1][i + 1],
            localMem[j + 1][i - 1], localMem[j - 1][i - 1], FLUX_FN);
#endif

        img[gx * info.strides[0] + gy * info.strides[1]] = (T)(C + delta * dt);
    }
}
