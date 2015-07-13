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
#include <vector>
#include <string>
#include <af/util.h>
#include <math.h>
#include "mnist_common.h"

using namespace af;
using std::vector;

float accuracy(const array& predicted, const array& target)
{
    array val, plabels, tlabels;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// Derivative of the activation function
array deriv(const array &out)
{
    return out * (1 - out);
}

// Cost function
double error(const array &out,
             const array &pred)
{
    array dif = (out - pred);
    return sqrt((double)(sum<float>(dif * dif)));
}

array binary(const array in)
{
    // Choosing "1" with probability sigmoid(in)
    return (in > randu(in.dims())).as(f32);
}

class rbm {

private:
    array weights;
    array h_bias;
    array v_bias;

    // Add bias input to the output from previous layer
    array vtoh(const array &v)
    {
        return binary(prop_up(v));
    }

    array htov(const array &h)
    {
        return binary(prop_down(h));
    }

public:

    rbm() {}

    rbm(int v_size, int h_size) :
        weights(randu(h_size, v_size)/100 - 0.05),
        h_bias(constant(0, 1, h_size)),
        v_bias(constant(0, 1, v_size))
    {
    }

    array prop_up(const array &v)
    {
        array h_bias_tile = tile(h_bias, v.dims(0));
        return sigmoid(h_bias_tile + matmulNT(v, weights));
    }

    array prop_down(const array &h)
    {
        array v_bias_tile = tile(v_bias, h.dims(0));
        return sigmoid(v_bias_tile + matmul(h, weights));
    }

    void gibbs_vhv(array &vt, array &ht, const array &v, int k = 1)
    {
        vt = v;
        for (int i = 0; i < k; i++) {
            ht = vtoh(vt);
            vt = htov(ht);
        }
    }

    void gibbs_hvh(array &vt, array &ht, const array &h, int k = 1)
    {
        ht = h;
        for (int i = 0; i < k; i++) {
            vt = htov(ht);
            ht = vtoh(vt);
        }
    }

    void train(const array &in,
               double lr = 0.1,
               int num_epochs = 15,
               int batch_size = 100,
               int k = 1, bool verbose = false)
    {
        const int num_samples = in.dims(0);
        const int num_batches = num_samples / batch_size;

        for (int i = 0; i <  num_epochs; i++) {

            double err = 0;

            for (int j = 0; j < num_batches - 1; j++) {

                int st = j * batch_size;
                int en = std::min(num_samples - 1, st + batch_size - 1);
                int num = en - st + 1;

                array v_pos = in(seq(st, en), span);

                array h_pos = vtoh(v_pos);

                array v_neg, h_neg;

                gibbs_hvh(v_neg, h_neg, h_pos, k);

                // Update weights
                array c_pos = matmulTN(h_pos, v_pos);
                array c_neg = matmulTN(h_neg, v_neg);

                array delta_w = lr * (c_pos - c_neg) / num;
                array delta_vb = lr * sum(v_pos - v_neg) / num;
                array delta_hb = lr * sum(h_pos - h_neg) / num;

                weights += delta_w;
                v_bias += delta_vb;
                h_bias += delta_hb;

                if (verbose) {
                    err += error(v_pos, v_neg);
                }
            }

            if (verbose) {
                printf("Epoch %d: Reconstruction error: %0.4f\n", i + 1, err / num_batches);
            }
        }

        if (verbose) printf("\n");
    }
};

int rbm_demo(bool console, int perc)
{
    printf("** ArrayFire RBM Demo **\n\n");

    array train_images, test_images;
    array train_target, test_target;
    int num_classes, num_train, num_test;

    // Load mnist data
    float frac = (float)(perc) / 100.0;
    setup_mnist<true>(&num_classes, &num_train, &num_test,
                      train_images, test_images, train_target, test_target, frac);

    dim4 dims = train_images.dims();

    int feature_size = train_images.elements() / num_train;

    // Reshape images into feature vectors
    array train_feats = moddims(train_images, feature_size, num_train).T();
    array test_feats  = moddims(test_images , feature_size, num_test ).T();

    train_target = train_target.T();
    test_target  = test_target.T();

    rbm network(train_feats.dims(1), 2000);

    network.train(train_feats,
                  0.1, // learning rate
                  15,  // num epochs
                  100, // batch size
                  1,   // k
                  true);

    // Test reconstructed images
    for (int ii = 0; ii < 5; ii++) {

        array in = test_feats(ii, span);
        array res, tmp;

        network.gibbs_vhv(res, tmp, in);

        in  = moddims(in , dims[0], dims[1]);
        res = moddims(res, dims[0], dims[1]);

        in = round(in);
        res = round(res);

        printf("Reconstructed Error for image %2d: %.4f\n", ii,
               sum<float>(abs(in - res)) / feature_size);
    }

    return 0;
}

int main(int argc, char** argv)
{
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 60;

    try {

        af::setDevice(device);
        af::info();
        return rbm_demo(console, perc);

    } catch (af::exception &ae) {
        std::cerr << ae.what() << std::endl;
    }

}
