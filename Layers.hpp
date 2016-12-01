//
//  Layers.hpp
//  face_rec
//
//  Created by 徐鹏 on 11/2/16.
//  Copyright © 2016 徐鹏. All rights reserved.
//

#ifndef Layers_hpp
#define Layers_hpp
#include "Neuron.hpp"
#include <stdio.h>
class Layer
{
private:
    static const int Neuron_max_num = 10;
public:
    Layer();
    Layer(int neurons_num);
    ~Layer();
    Neuron neurons_[Neuron_max_num];
    int neurons_num_;
};

#endif /* Layers_hpp */
