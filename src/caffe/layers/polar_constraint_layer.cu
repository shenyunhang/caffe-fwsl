#include <vector>

#include "caffe/layers/polar_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // data 	#roi 	#class 	1 	1
  // filter_	#roi 	#class 	1 	1
  // [-1,1] denote the score

  caffe_copy(count_, bottom[1]->gpu_data(), filter_.mutable_gpu_data());
  if (polar_) {
    // minima is 0
    caffe_gpu_threshold(count_, filter_.gpu_data(), filter_.mutable_gpu_data(),
                        Dtype(0), Dtype(0), false);
  } else {
    // maxima is 0
    caffe_gpu_threshold(count_, filter_.gpu_data(), filter_.mutable_gpu_data(),
                        Dtype(0), Dtype(0), true);
    caffe_gpu_scal(count_, Dtype(-1), filter_.mutable_gpu_data());
  }
  caffe_gpu_mul(count_, filter_.gpu_data(), bottom[0]->gpu_data(),
                top[0]->mutable_gpu_data());
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to second inputs.";
  }

  if (propagate_down[0]) {
    caffe_gpu_mul(count_, filter_.gpu_data(), top[0]->gpu_diff(),
                  bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PolarConstraintLayer);

}  // namespace caffe
