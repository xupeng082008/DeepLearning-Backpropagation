//
//  Neuron.cpp
//  face_rec
//
//  Created by 徐鹏 on 11/2/16.
//  Copyright © 2016 徐鹏. All rights reserved.
//

#include "Neuron.hpp"
#include <math.h>

Neuron::Neuron()
{
    neuron_net_input_ = 0.;
}
Neuron::Neuron(double input)
{
    neuron_net_input_ = input;
}


Neuron::~Neuron()
{
}

Neuron::Neuron(int neurons_input_num) :neuron_input_num_(neurons_input_num)
{
}


double Neuron::calculate_total_net_input(double* layer_weights,
                                         double* neuron_inputs,
                                         int neuron_input_num)
{
    for (size_t i = 0; i < neuron_input_num+1; i++)
    {
        neuron_inputs_[i] = neuron_inputs[i];
        neuron_weights_[i] = layer_weights[i];
    }
    neuron_net_input_ = 0.;//***************very import**********************
    for (size_t i = 0; i < neuron_input_num + 1; i++)
        neuron_net_input_ += neuron_weights_[i] * neuron_inputs_[i];//z=wx+b
    return neuron_net_input_;
}

void Neuron::set_neuron_value(double input)
{
    neuron_net_input_ = input;
}
double Neuron::get_neuron_value(double input)
{
    neuron_net_input_ = input;
    return neuron_net_input_;
}

double Neuron::calculate_output(double net_input)
{
    neuron_net_output_ = activite_fun(net_input);
    return neuron_net_output_;
}

double Neuron::activite_fun(double net_input)
{
    return 1 / (1 + exp(-net_input));
}

double Neuron::calculate_activite_fun_diff(double fun_value)
{
    return fun_value * (1 - fun_value);
}