//
//  Backpropagation.hpp
//  face_rec
//
//  Created by 徐鹏 on 11/2/16.
//  Copyright © 2016 徐鹏. All rights reserved.
//
#pragma once
#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include "Layers.hpp"
#include "Neuron.hpp"
class NeuralNetwork
{
public:
    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(int layers_num, int inputlayer_neurons_num, int hiddenlayer_neurons_num, int outputlayer_neurons_num,
                  double* hidden_layer_weights, double* output_layer_weights,
                  double*  net_inputs, double* target_outs,double* bias);
    int train();
    void forward_pass();
    void back_propagate();
    double calculate_total_error();
	std::vector<double> predict(const std::vector<double>innputs);
    double learning_rate = 0.5;
    
private:
    static const int LAYERS_MAX_NUM = 10;
    static const int NET_INPUT_MAX_NUM = 10;
    static const int NET_OUTPUT_MAX_NUM = 10;
    static const int HIDDEN_WEIGHTS_MAX_NUM=100;
    static const int OUTPUT_WEIGHTS_MAX_NUM=10;
    int inputlayer_neurons_num_, hiddenlayer_neurons_num_, outputlayer_neurons_num_, layers_num_;
    Layer layers_[LAYERS_MAX_NUM];
    
    double net_inputs_[NET_INPUT_MAX_NUM];
    double target_outs_[NET_OUTPUT_MAX_NUM];
    
    double hidden_layer_weights_[HIDDEN_WEIGHTS_MAX_NUM];
    double output_layer_weights_[HIDDEN_WEIGHTS_MAX_NUM];
    double bias_[LAYERS_MAX_NUM];
    double total_error_;
};


#endif /* NeuralNetwork_hpp */
