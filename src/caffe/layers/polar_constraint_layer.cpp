#include <vector>

#include "caffe/layers/polar_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PolarConstraintLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  // data 	#roi 	#class 	1 	1
  // filter_	#roi 	#class 	1 	1
  // [-1,1] denote the score

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "POLAR_CONSTRAINT layer inputs must have the same shape.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
      << "POLAR_CONSTRAINT layer inputs must have the same shape.";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "POLAR_CONSTRAINT layer inputs must have the same count.";

  polar_ = this->layer_param_.polar_constraint_param().polar();
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  filter_.ReshapeLike(*bottom[0]);
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(PolarConstraintLayer, Backward);
#endif

INSTANTIATE_CLASS(PolarConstraintLayer);
REGISTER_LAYER_CLASS(PolarConstraint);

}  // namespace caffe
