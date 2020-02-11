/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <stdio.h>
#include <af/util.h>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace af;

array distance(array data, array means) {
    int n = data.dims(0);   // Number of features
    int k = means.dims(1);  // Number of means

    array data2  = tile(data, 1, k, 1);
    array means2 = tile(means, n, 1, 1);

    // Currently using manhattan distance
    // Can be replaced with other distance measures
    return sum(abs(data2 - means2), 2);
}

// Get cluster id of each location in data
array clusterize(const array data, const array means) {
    // Get manhattan distance
    array dists = distance(data, means);

    // get the locations of minimum distance
    array idx, val;
    min(val, idx, dists, 1);

    // Return cluster IDs
    return idx;
}

array new_means(array data, array clusters, int k) {
    int d           = data.dims(2);
    array means     = constant(0, 1, k, d);
    array clustersd = tile(clusters, 1, 1, d);

    gfor(seq ii, k) {
        means(span, ii, span) =
            sum(data * (clustersd == ii)) / (sum(clusters == ii) + 1e-5);
    }

    return means;
}

// kmeans(means, clusters, data, k)
// data:  input,  1D or 2D (range > [0-1])
// k:     input,  # desired means (k > 1)
// means: output, vector of means
void kmeans(array &means, array &clusters, const array in, int k,
            int iter = 100) {
    unsigned n = in.dims(0);  // Num features
    unsigned d = in.dims(2);  // feature length

    // reshape input
    array data = in * 0;

    // re-center and scale down data to [0, 1]
    array minimum = min(in);
    array maximum = max(in);

    gfor(seq ii, d) {
        data(span, span, ii) =
            (in(span, span, ii) - minimum(ii).scalar<float>()) /
            maximum(ii).scalar<float>();
    }

    // Initial guess of means
    means               = randu(1, k, d);
    array curr_clusters = constant(0, data.dims(0)) - 1;
    array prev_clusters;

    // Stop updating after specified number of iterations
    for (int i = 0; i < iter; i++) {
        // Store previous cluster ids
        prev_clusters = curr_clusters;

        // Get cluster ids for current means
        curr_clusters = clusterize(data, means);

        // Break early if clusters not changing
        unsigned num_changed = count<unsigned>(prev_clusters != curr_clusters);

        if (num_changed < (n / 1000) + 1) break;

        // Update current means for new clusters
        means = new_means(data, curr_clusters, k);
    }

    // Scale up means
    gfor(seq ii, d) {
        means(span, span, ii) =
            maximum(ii) * means(span, span, ii) + minimum(ii);
    }

    clusters = prev_clusters;
}

// K-Means image recoloring.
// Shifts the hues of an image to the k mean hues.
int kmeans_demo(int k, bool console) {
    printf("** ArrayFire K-Means Demo (k = %d) **\n\n", k);

    array img =
        loadImage(ASSETS_DIR "/examples/images/spider.jpg") / 255;  // [0-255]

    int w = img.dims(0), h = img.dims(1), c = img.dims(2);
    array vec = moddims(img, w * h, 1, c);

    array means_full, clusters_full;
    kmeans(means_full, clusters_full, vec, k);

    array means_half, clusters_half;
    kmeans(means_half, clusters_half, vec, k / 2);

    array means_dbl, clusters_dbl;
    kmeans(means_dbl, clusters_dbl, vec, k * 2);

    if (!console) {
        array out_full =
            moddims(means_full(span, clusters_full, span), img.dims());
        array out_half =
            moddims(means_half(span, clusters_half, span), img.dims());
        array out_dbl =
            moddims(means_dbl(span, clusters_dbl, span), img.dims());

        af::Window wnd(800, 800, "ArrayFire K-Means Demo");
        wnd.grid(2, 2);
        std::stringstream out_full_caption, out_half_caption, out_dbl_caption;
        out_full_caption << "k = " << k;
        out_half_caption << "k = " << k / 2;
        out_dbl_caption << "k = " << k * 2;
        while (!wnd.close()) {
            wnd(0, 0).image(img, "Input Image");
            wnd(0, 1).image(out_full, out_full_caption.str().c_str());
            wnd(1, 0).image(out_half, out_half_caption.str().c_str());
            wnd(1, 1).image(out_dbl, out_dbl_caption.str().c_str());
            wnd.show();
        }
    } else {
        means_full =
            moddims(means_full, means_full.dims(1), means_full.dims(2));
        means_half =
            moddims(means_half, means_half.dims(1), means_half.dims(2));
        means_dbl = moddims(means_dbl, means_dbl.dims(1), means_dbl.dims(2));

        af_print(means_full);
        af_print(means_half);
        af_print(means_dbl);
    }

    return 0;
}

int main(int argc, char **argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int k        = argc > 3 ? atoi(argv[3]) : 8;

    try {
        af::setDevice(device);
        af::info();
        return kmeans_demo(k, console);

    } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }

    return 0;
}
