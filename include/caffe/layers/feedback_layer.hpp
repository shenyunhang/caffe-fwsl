#ifndef CAFFE_FEEDBACK_LAYER_HPP_
#define CAFFE_FEEDBACK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class FeedbackLayer : public Layer<Dtype> {
 public:
  explicit FeedbackLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Feedback"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  // const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  // const vector<bool>& propagate_down,
  // const vector<Blob<Dtype>*>& bottom);

  void Show_blob(const Dtype* data, const int channels, const int height,
                 const int width, const string save_path,
                 const string save_path_jet, const float threshold_ratio,
                 const int fill = 0);
  void vis_det(const Blob<Dtype>* det_blob, const Blob<Dtype>* img_blob,
               Blob<Dtype>& score_map, int save_id_);

  int num_img_;
  int num_det_;

  Blob<Dtype> score_map_;
  int save_id_;
};

}  // namespace caffe

#endif  // CAFFE_FEEDBACK_LAYER_HPP_
