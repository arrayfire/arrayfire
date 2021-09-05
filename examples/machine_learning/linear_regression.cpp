
/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Demo of ordinary least squares Linear Regression on the NCSU diabetes dataset.
// Fits a linear model with coefficients w = (w1, …, wp) 
// to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
// Note : Not using the original dataset from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
// but the normalized dataset : 
// https://www4.stat.ncsu.edu/~boos/var.select/diabetes.read.rdata.out.txt
// where the x variables have been standardized to have mean 0 and squared length = 1 (sum(x^2)=1). 
// Each of these 10 feature variables have been mean centered and scaled by the standard deviation times n_samples (i.e. the sum of squares of each column totals 1).
// Reproduces the results of scikit-learn:
// https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
// Expected results:
// Regression coeffs: Slope: 938 Intercept: 152
// Mean Squared Error: 2548
// Coefficient of determination (r2): 0.47

#include <arrayfire.h>
#include <math.h>

#include <stdio.h>
#include <af/util.h>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace af;

// Parse a line of scalars and insert the values in the given vector
template<typename T>
void parseFloatLine(std::ifstream& ifs, std::vector<T>& v) {
    v.clear();
    std::string line;
    getline(ifs, line);
    std::stringstream ss(line);
    T f;
    while(!ss.eof()) {
        ss >> f;
	v.push_back(f);
    }
}

int demo(bool console, const int perc) {
    // Check with reviewers : push the data file to the AF assets repository ?
    const std::string filepath = "../examples/machine_learning/diabetes.read.rdata.out.txt"; 
    std::ifstream ifs(filepath.c_str());
    if (!ifs.good()) {
        std::cerr << "Cannot open dataset file " << filepath << std::endl;
        return -1;
    }
    std::string line;
    std::getline(ifs, line); // just ignore the header line

    ifs.unsetf(std::ios_base::skipws);
    const unsigned N = std::count(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), '\n');
    if (N == 0) {
        std::cerr << "Cannot read any line of data from the dataset file" << std::endl;
        return -1;
    }

    ifs.clear();
    ifs.seekg(0, std::ios::beg); // rewind
    std::getline(ifs, line); // just ignore the header line

    std::vector<float> trainX, testX; // patients ages (independent variable)
    std::vector<float> trainY, testY; // diabete disease progression: "quantitative measure of disease progression one year after baseline" (dependent variable)
    std::vector<float> row;
    unsigned n = 0;
    float frac = (float)(perc) / 100.0f; // fraction of N to be used for training vs testing
    while (!ifs.eof()) {
	parseFloatLine(ifs, row);
	if (row.size() < 11)
            break;
	++n;
        // column 3 is the bmi (blood mass index), column 11 is Y (diabete progression)  
        if (n < frac*N) {
	    trainX.push_back(row[3]);
	    trainY.push_back(row[11]);
        } else {
	    testX.push_back(row[3]);
	    testY.push_back(row[11]);
        }
    }

    if (trainY.size() + testY.size() != N) {
        std::cerr << "Unexpected number of datapoints: " << trainY.size() << " + " << testY.size()  << " vs " << N << std::endl;
        return -1;
    }
    std::cout << N << " data points extracted : " << trainX.size() << " for training, " << testX.size() << " for testing.\n";

    af::array trainXarray(trainX.size(), trainX.data());
    af::array trainYarray(trainY.size(), trainY.data());

    af::array testXarray(testX.size(), testX.data());
    af::array testYarray(testY.size(), testY.data());

    float meanX = af::mean<float>(trainXarray);
    float meanY = af::mean<float>(trainYarray);
    float meanX2 = af::mean<float>(trainXarray*trainXarray);
    float meanY2 = af::mean<float>(trainYarray*trainYarray);
    float meanXY = af::mean<float>(trainXarray*trainYarray);

    float SSxy = af::sum<float>((trainXarray-meanX)*(trainYarray-meanY));
    float SSxx = af::sum<float>(af::pow(trainXarray-meanX, 2));

    // slope = B1 = SSxy / SSxx
    float slope = SSxy / SSxx;
    std::cout << "Coefficients: " << slope << std::endl;
    // intercept = B0 = meanY - B1*meanX
    float intercept = meanY - slope*meanX;
    std::cout << "Intercept: " << intercept << std::endl;

    af::array predictedY = testXarray*slope + intercept;
    float RSS = af::sum<float>(af::pow(predictedY - testYarray, 2)); // the residual sum of squares
    float MSE = RSS / testY.size();
    std::cout << "Mean squared error: " << MSE << std::endl;

    // Coefficient of determination R2 = 1 - RSS / SST  with SST is the total sum of squared 
    float SST = af::sum<float>(af::pow(testYarray - af::mean<float>(testYarray), 2));
    float r2 = 1 - RSS / SST; 
    std::cout << "Coefficient of determination (r2) = " << r2 << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 95; // percentage training/test
    std::cout << argv[0] << std::endl;

    try {
        af::setDevice(device);
        af::info();
        return demo(console, perc);
    } catch (af::exception &ae) { 
        std::cerr << ae.what() << std::endl; 
    }

    return 0;
}

