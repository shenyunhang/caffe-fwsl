#ifndef CAFFE_MIL_LAYER_HPP_
#define CAFFE_MIL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/cpg_layer.hpp"
#include "caffe/layers/repartition_layer.hpp"

namespace caffe {

template <typename Dtype>
class MILLayer : public Layer<Dtype> {
 public:
  explicit MILLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        cpg_layer_(new CPGLayer<Dtype>(param)),
        repartition_layer_(new RepartitionLayer<Dtype>(param)) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MIL"; }
  // virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MaxNumBottomBlobs() const { return 5; }
  virtual inline int MinNumBottomBlobs() const { return 3; }
  // virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 3; }
  virtual inline int MinNumTopBlobs() const { return 1; }

  void Set_Net(Net<Dtype>* net) {
    LOG(INFO) << "Setting net in MILLayer";
    net_ = net;
  }

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

  Net<Dtype>* net_;

  map<string, int> bottom_index_;
  map<string, int> top_index_;

  shared_ptr<CPGLayer<Dtype> > cpg_layer_;
  vector<Blob<Dtype>*> cpg_bottom_vec_;
  vector<Blob<Dtype>*> cpg_top_vec_;

  shared_ptr<RepartitionLayer<Dtype> > repartition_layer_;
  vector<Blob<Dtype>*> repartition_bottom_vec_;
  vector<Blob<Dtype>*> repartition_top_vec_;

  Blob<Dtype> cpg_blob_;
};

}  // namespace caffe

#endif  // CAFFE_MIL_LAYER_HPP_
