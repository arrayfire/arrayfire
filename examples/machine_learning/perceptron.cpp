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

float accuracy(const array& predicted, const array& target)
{
    array val, plabels, tlabels;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);

    return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// Predict based on given parameters
array predict(const array &X, const array &Weights)
{
    return sigmoid(matmul(X, Weights));
}

array train(const array &X, const array &Y,
            double alpha = 0.1,
            double maxerr = 0.05,
            int maxiter = 1000, bool verbose = false)
{

    // Initialize parameters to 0
    array Weights = constant(0, X.dims(1), Y.dims(1));

    for (int i = 0; i < maxiter; i++) {
        array P = predict(X, Weights);
        array err = Y - P;

        float mean_abs_err = mean<float>(abs(err));
        if (mean_abs_err  < maxerr) break;

        if (verbose && (i + 1) % 25 == 0) {
            printf("Iter: %d, Err: %.4f\n", i + 1, mean_abs_err);
        }

        Weights = Weights + alpha * matmulTN(X, err);
    }

    return Weights;
}

void benchmark_perceptron(const array &train_feats,
                          const array &train_targets,
                          const array test_feats)
{
    timer::start();
    array Weights = train(train_feats, train_targets, 0.1, 0.01, 1000);
    af::sync();
    printf("Training time: %4.4lf s\n", timer::stop());

    timer::start();
    const int iter = 100;
    for (int i = 0; i < iter; i++) {
        array test_outputs  = predict(test_feats , Weights);
        test_outputs.eval();
    }
    af::sync();
    printf("Prediction time: %4.4lf s\n", timer::stop() / iter);
}

// Demo of one vs all logistic regression
int perceptron_demo(bool console, int perc)
{
    array train_images, train_targets;
    array test_images, test_targets;
    int num_train, num_test, num_classes;

    // Load mnist data
    float frac = (float)(perc) / 100.0;
    setup_mnist<true>(&num_classes, &num_train, &num_test,
                      train_images, test_images,
                      train_targets, test_targets, frac);

    // Reshape images into feature vectors
    int feature_length = train_images.elements() / num_train;
    array train_feats = moddims(train_images, feature_length, num_train).T();
    array test_feats  = moddims(test_images , feature_length, num_test ).T();

    train_targets = train_targets.T();
    test_targets  = test_targets.T();

    // Add a bias that is always 1
    train_feats = join(1, constant(1, num_train, 1), train_feats);
    test_feats  = join(1, constant(1, num_test , 1), test_feats );

    // Train logistic regression parameters
    array Weights = train(train_feats, train_targets, 0.1, 0.01, 1000, true);

    // Predict the results
    array train_outputs = predict(train_feats, Weights);
    array test_outputs  = predict(test_feats , Weights);

    printf("Accuracy on training data: %2.2f\n",
           accuracy(train_outputs, train_targets ));

    printf("Accuracy on testing data: %2.2f\n",
           accuracy(test_outputs , test_targets ));

    benchmark_perceptron(train_feats, train_targets, test_feats);

    if (!console) {
        test_outputs = test_outputs.T();
        test_targets = test_targets.T();
        // Get 20 random test images.
        display_results<true>(test_images, test_outputs, test_targets, 20);
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
        return perceptron_demo(console, perc);

    } catch (af::exception &ae) {
        std::cerr << ae.what() << std::endl;
    }

}
