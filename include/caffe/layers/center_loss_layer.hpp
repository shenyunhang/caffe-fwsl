#ifndef CAFFE_CENTER_LOSS_LAYER_HPP_
#define CAFFE_CENTER_LOSS_LAYER_HPP_

#include <vector>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CenterLossLayer : public LossLayer<Dtype> {
 public:
  explicit CenterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  // bottom:
  // 1: feature
  // 2: label
  // 3: bbox score

  // We repalce the center_ with blob_.
  // So that the center will be saved when the net is saved.
  // However, we don't update the center by the solver, but by this layer
  // itself.
  // So the learning rate of this layer need set to zero.
  // Blob<Dtype> center_;

  bool debug_info_;
  int num_center_;
  int top_k_;
  int update_;
  Dtype lr_;

  int dim_;
  int num_class_;

  Blob<Dtype> diff_;
  int num_gt_class_;
  int num_roi_;
  vector<set<int> > roi_sets_;
  vector<int> center_selector_;

  vector<vector<int> > num_update_class_;

  // output for visualization
  Dtype accum_loss_;
  int total_iter_;
  vector<vector<int> > accum_update_class_;

  bool is_center_;
  int display_;
  int pass_im_;
  int max_num_im_center_;
};

}  // namespace caffe

#endif  // CAFFE_CENTER_LOSS_LAYER_HPP_
