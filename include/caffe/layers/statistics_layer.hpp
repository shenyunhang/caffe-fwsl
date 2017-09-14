#ifndef CAFFE_STATISTICS_LAYER_HPP_
#define CAFFE_STATISTICS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class StatisticsLayer : public Layer<Dtype> {
 public:
  explicit StatisticsLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Statistics"; }
  virtual inline int ExactNumBottomBlobs() const { return 5; }
  // virtual inline int MaxBottomBlobs() const { return 5; }
  // virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  // const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  // const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  map<string, int> bottom_index_;
  vector<string> voc_label_;
  int display_;
  Dtype predict_threshold_;
  int num_class_;
  int num_im_;
  int num_roi_;

  int pass_im_;

  struct statistic {
    vector<int> label;
    vector<Dtype> predict;
    vector<Dtype> predict_pos;
    vector<Dtype> predict_neg;
    vector<int> roi;
    vector<int> roi_left;

    int accum_label;
    Dtype accum_predict;
    Dtype accum_predict_pos;
    Dtype accum_predict_neg;
    int accum_roi;
    int accum_roi_left;
  };

  statistic ori_;
  statistic cpg_;

};

}  // namespace caffe

#endif  // CAFFE_STATISTICS_LAYER_HPP_
