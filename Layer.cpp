#include "Layer.h"


Layer::Layer()
{
}


Layer::~Layer()
{
	if (NULL != neurons_){
		delete[] neurons_;
		neurons_ = NULL;
	}
	if (NULL != neurons_){
		delete[] neurons_;
		neurons_ = NULL;
	}

}

Layer::Layer(int neurons_num) : neurons_num_(neurons_num)
{
	neurons_ = new Neuron[neurons_num_];
}
