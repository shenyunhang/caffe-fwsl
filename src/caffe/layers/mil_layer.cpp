#include <vector>

#include "caffe/layers/mil_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void deviceQuery() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  // for (dev = 0; dev < deviceCount; ++dev) {
  // cudaSetDevice(dev);
  cudaGetDevice(&dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

  // Console log
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
         driverVersion / 1000, (driverVersion % 100) / 10,
         runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
         deviceProp.major, deviceProp.minor);
  //}
}

template <typename Dtype>
void MILLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  bottom_index_["label"] = 0;
  bottom_index_["predict"] = 1;
  bottom_index_["rois"] = 2;
  bottom_index_["rois_score"] = 3;
  bottom_index_["filter"] = 4;
  bottom_index_["io"] = 5;

  top_index_["select"] = 1;
  top_index_["poslabel"] = 2;
  top_index_["neglabel"] = 3;

  // LayerParameter mil_param(this->layer_param_);

  // TODO(YH): we can directly create new layers without using registry
  // mil_param.set_type("CPG");
  // cpg_layer_ = LayerRegistry<Dtype>::CreateLayer(mil_param);
  // shared_ptr<CPGLayer<Dtype> > cpg_layer__ =
  // boost::dynamic_pointer_cast<CPGLayer<Dtype> >(cpg_layer_);
  // cpg_layer__->Set_Net(net_);
  cpg_layer_->Set_Net(net_);
  cpg_bottom_vec_.clear();
  cpg_bottom_vec_.push_back(bottom[bottom_index_["label"]]);
  //cpg_bottom_vec_.push_back(bottom[bottom_index_["predict"]]);
  cpg_top_vec_.clear();
  cpg_top_vec_.push_back(&cpg_blob_);
  cpg_layer_->SetUp(cpg_bottom_vec_, cpg_top_vec_);

  // TODO(YH): we can directly create new layers without using registry
  // mil_param.set_type("Repartition");
  // repartition_layer_ = LayerRegistry<Dtype>::CreateLayer(mil_param);
  // repartition_layer_ = new RepartitionLayer<Dtype>(mil_param);
  repartition_bottom_vec_.clear();
  repartition_bottom_vec_.push_back(&cpg_blob_);
  repartition_bottom_vec_.push_back(bottom[bottom_index_["label"]]);
  repartition_bottom_vec_.push_back(bottom[bottom_index_["predict"]]);
  repartition_bottom_vec_.push_back(bottom[bottom_index_["rois"]]);
  repartition_bottom_vec_.push_back(bottom[bottom_index_["rois_score"]]);
  if (bottom.size() > bottom_index_["filter"]) {
    repartition_bottom_vec_.push_back(bottom[bottom_index_["filter"]]);
    repartition_bottom_vec_.push_back(bottom[bottom_index_["io"]]);
  }
  repartition_top_vec_.clear();
  for (size_t i = 0; i < top.size(); ++i) {
    repartition_top_vec_.push_back(top[i]);
  }
  repartition_layer_->SetUp(repartition_bottom_vec_, repartition_top_vec_);

  // get the cuda version
  LOG(INFO) << "-------------------------------------------------------------";
  LOG(INFO) << "Show current device and CUDA info";
  deviceQuery();
  LOG(INFO) << "-------------------------------------------------------------";
}

template <typename Dtype>
void MILLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
  cpg_layer_->Reshape(cpg_bottom_vec_, cpg_top_vec_);
  repartition_layer_->Reshape(repartition_bottom_vec_, repartition_bottom_vec_);
}

template <typename Dtype>
void MILLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MILLayer);
#endif

INSTANTIATE_CLASS(MILLayer);
REGISTER_LAYER_CLASS(MIL);

}  // namespace caffe
