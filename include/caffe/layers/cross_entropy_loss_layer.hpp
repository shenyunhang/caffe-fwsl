#ifndef CAFFE_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

const float kRECIPROCAL_THRESHOLD = 1e-4;
const float kDIFF_THRESHOLD = 1e+4;

template <typename Dtype>
class CrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit CrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrossEntropyLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int display_;
  int ignore_label_;
  int ignore_value_;

  Dtype accum_loss_;
  int accum_iter_;
  int accum_ignore_num_;

  int count_;
  int num_im_;
  int num_class_;

};

}  // namespace caffe

#endif  // CAFFE_CROSS_ENTROPY_LOSS_LAYER_HPP_
