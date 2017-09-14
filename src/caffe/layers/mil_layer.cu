#include <vector>

#include "caffe/layers/mil_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MILLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  cpg_layer_->Forward(cpg_bottom_vec_, cpg_top_vec_);
  repartition_layer_->set_gt_class(cpg_layer_->get_gt_class());
  repartition_layer_->Forward(repartition_bottom_vec_, repartition_top_vec_);
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MILLayer);

}  // namespace caffe
