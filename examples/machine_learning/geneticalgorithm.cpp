/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <ctime>

using namespace af;
static const float DefaultTopFittest = 0.5;

array update(const array& searchSpace, const array& sampleX,
             const array& sampleY, const int n) {
    return searchSpace(sampleY * n + sampleX);
}

array selectFittest(const array& sampleZ, const int nSamples,
                    const float topFit = DefaultTopFittest) {
    // pick top fittest
    array indices, values;
    sort(values, indices, sampleZ);
    int topFitElem = topFit * nSamples;
    int n          = indices.elements();
    return (n > topFitElem) ? indices(seq(n - topFitElem, n - 1)) : indices;
}

void reproduce(array& searchSpace, array& sampleX, array& sampleY,
               array& sampleZ, const int nSamples, const int n) {
    // Get fittest parents
    array selection = selectFittest(sampleZ, nSamples);
    array parentsX  = sampleX(selection);
    array parentsY  = sampleY(selection);
    int bits        = (int)log2(n);

    // Divide selection in two
    array parentsX1 = parentsX.rows(0, parentsX.elements() / 2 - 1);
    array parentsX2 =
        parentsX.rows(parentsX.elements() / 2, parentsX.elements() - 1);
    array parentsY1 = parentsY.rows(0, parentsY.elements() / 2 - 1);
    array parentsY2 =
        parentsY.rows(parentsY.elements() / 2, parentsY.elements() - 1);

    // Get crossover points (at which bit to crossover) and construct bit masks
    // from them
    array crossover = randu(nSamples / 4, u32) % bits;
    array lowermask = (1 << crossover) - 1;
    array uppermask = INT_MAX - lowermask;

    // Create children as the cross between two parents
    array childrenX1 = (parentsX1 & uppermask) + (parentsX2 & lowermask);
    array childrenY1 = (parentsY1 & uppermask) + (parentsY2 & lowermask);

    array childrenX2 = (parentsX2 & uppermask) + (parentsX1 & lowermask);
    array childrenY2 = (parentsY2 & uppermask) + (parentsY1 & lowermask);

    // Join two new sets
    sampleX = join(0, childrenX1, childrenX2);
    sampleY = join(0, childrenY1, childrenY2);

    // Create mutant children
    array mutantX = sampleX;
    array mutantY = sampleY;

    // Flip a random bit to vary the gene pool
    mutantX = mutantX ^ (1 << (randu(nSamples / 2, u32) % bits));
    mutantY = mutantY ^ (1 << (randu(nSamples / 2, u32) % bits));

    sampleX = join(0, sampleX, mutantX);
    sampleY = join(0, sampleY, mutantY);

    // Update the value of each sample with the new coordinates
    sampleZ = update(searchSpace, sampleX, sampleY, n);
}

void initSamples(array& searchSpace, array& sampleX, array& sampleY,
                 array& sampleZ, const int nSamples, const int n) {
    setSeed(time(NULL));
    sampleX = randu(nSamples, u32) % n;
    sampleY = randu(nSamples, u32) % n;
    sampleZ = update(searchSpace, sampleX, sampleY, n);
}

void init(array& searchSpace, array& searchSpaceXDisplay,
          array& searchSpaceYDisplay, array& sampleX, array& sampleY,
          array& sampleZ, const int nSamples, const int n) {
    // initialize space
    searchSpace = range(dim4(n / 2, n / 2), 0) + range(dim4(n / 2, n / 2), 1);
    searchSpace = join(0, searchSpace, flip(searchSpace, 0));
    searchSpace = join(1, searchSpace, flip(searchSpace, 1));

    // initialize display data
    searchSpaceXDisplay = iota(dim4(n, 1), dim4(1, n));
    searchSpaceYDisplay = iota(dim4(1, n), dim4(n, 1));

    // initalize searchers
    initSamples(searchSpace, sampleX, sampleY, sampleZ, nSamples, n);
}

void reproducePrint(float& currentMax, array& searchSpace, array& sampleX,
                    array& sampleY, array& sampleZ, const float trueMax,
                    const int nSamples, const int n) {
    if (currentMax < trueMax * 0.99) {
        float maximum = max<float>(sampleZ);
        array whereM  = where(sampleZ == maximum);
        if (maximum < trueMax * 0.99) {
            printf("Current max at ");
        } else {
            printf("\nMax found at ");
        }
        printf("(%d,%d): %f (trueMax %f)\n",
               sampleX(whereM).scalar<unsigned int>(),
               sampleY(whereM).scalar<unsigned int>(), maximum, trueMax);
        currentMax = maximum;
        reproduce(searchSpace, sampleX, sampleY, sampleZ, nSamples, n);
    }
}

void geneticSearch(bool console, const int nSamples, const int n) {
    array searchSpaceXDisplay = 0;
    array searchSpaceYDisplay = 0;
    array searchSpace;
    array sampleX;
    array sampleY;
    array sampleZ;

    init(searchSpace, searchSpaceXDisplay, searchSpaceYDisplay, sampleX,
         sampleY, sampleZ, nSamples, n);
    float trueMax = max<float>(searchSpace);
    float maximum = -trueMax;

    if (!console) {
        af::Window win(1600, 800, "Arrayfire Genetic Algorithm Search Demo");
        win.grid(1, 2);
        do {
            reproducePrint(maximum, searchSpace, sampleX, sampleY, sampleZ,
                           trueMax, nSamples, n);
            win(0, 0).setAxesTitles("IdX", "IdY", "Search Space");
            win(0, 1).setAxesTitles("IdX", "IdY", "Search Space");
            win(0, 0).surface(searchSpaceXDisplay, searchSpaceYDisplay,
                              searchSpace);
            win(0, 1).scatter(sampleX.as(f32), sampleY.as(f32), sampleZ.as(f32),
                              AF_MARKER_CIRCLE);
            win.show();
        } while (!win.close());
    } else {
        do {
            reproducePrint(maximum, searchSpace, sampleX, sampleY, sampleZ,
                           trueMax, nSamples, n);
        } while (maximum < trueMax * 0.99);
    }
}

int main(int argc, char** argv) {
    bool console       = false;
    const int n        = 32;
    const int nSamples = 16;
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "-"))) {
        printf("usage: %s [-]\n", argv[0]);
        return -1;
    } else if (argc == 2 && argv[1][0] == '-') {
        console = true;
    }

    try {
        af::info();
        printf("** ArrayFire Genetic Algorithm Search Demo **\n\n");
        printf(
            "Search for trueMax in a search space where the objective function "
            "is defined as :\n\n");
        printf("SS(x ,y) = min(x, n - (x + 1)) + min(y, n - (y + 1))\n\n");
        printf("(x, y) belongs to RxR; R = [0, n); n = %d\n\n", n);
        if (!console) {
            printf("The left figure shows the objective function.\n");
            printf(
                "The figure on the right shows current generation's parameters "
                "and function values.\n\n");
        }
        geneticSearch(console, nSamples, n);
    } catch (af::exception& e) { fprintf(stderr, "%s\n", e.what()); }

    return 0;
}
