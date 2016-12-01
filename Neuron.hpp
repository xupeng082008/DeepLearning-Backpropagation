//
//  Neuron.hpp
//  face_rec
//
//  Created by 徐鹏 on 11/2/16.
//  Copyright © 2016 徐鹏. All rights reserved.
//
#pragma once
#ifndef Neuron_hpp
#define Neuron_hpp
#include <stdlib.h>
class Neuron
{
private:
    static const int NEURON_INPUT_MAX_NUM = 100;
    static const int NEURON_WEIGHTS_MAX_NUM = 100;
public:
    Neuron();
    ~Neuron();
    Neuron(double input);
    Neuron(int neurons_input_num);
    
    double calculate_total_net_input(double* layer_weights, double* neuron_inputs,
                                     int neuron_input_num);
    double calculate_output(double net_inputs);
    double activite_fun(double net_input);
    double calculate_activite_fun_diff(double fun_value);
    double get_neuron_value(double input);
    void set_neuron_value(double input);
    
    double neuron_net_input_;
    double neuron_net_output_;
    
    //private:
    int neuron_input_num_;
    double neuron_inputs_[NEURON_INPUT_MAX_NUM];
    double neuron_weights_[NEURON_WEIGHTS_MAX_NUM];
    double deltas_;

    
    
    
};
#endif /* Neuron_hpp */
