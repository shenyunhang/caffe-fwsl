#include <time.h>
#include <boost/filesystem.hpp>
#include <vector>

#include "caffe/layers/cpg_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
  // for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

  return buf;
}

template <typename Dtype>
void CPGLayer<Dtype>::Show_info() {
  if (!is_show_) return;
  is_show_ = false;

  LOG(INFO) << "==============================================";
  LOG(INFO) << "CPG layer:";
  LOG(INFO) << "is_cpg_: " << is_cpg_;
  LOG(INFO) << "ignore_label_: " << ignore_label_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "is_contrast: " << is_contrast_;
  LOG(INFO) << "predict_threshold_:" << predict_threshold_;
  LOG(INFO) << "start_layer_name_: " << start_layer_name_;
  LOG(INFO) << "start_layer_index_: " << start_layer_index_;
  LOG(INFO) << "end_layer_name_: " << end_layer_name_;
  LOG(INFO) << "end_layer_index_: " << end_layer_index_;
  for (size_t i = 0; i < cpg_blob_name_.size(); ++i) {
    LOG(INFO) << "cpg_blob_name_: " << cpg_blob_name_[i];
    LOG(INFO) << "cpg_blob_index_: " << cpg_blob_index_[i];
  }
  LOG(INFO) << "predict_blob_name_: " << predict_blob_name_;
  LOG(INFO) << "predict_blob_index_: " << predict_blob_index_;

  const vector<string> layer_names_ = net_->layer_names();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "show all layer names:";
  for (size_t i = 0; i < layer_names_.size(); i++)
    LOG(INFO) << i << ": " << layer_names_[i];

  const vector<string> blob_names_ = net_->blob_names();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "show all blob name:";
  for (size_t i = 0; i < blob_names_.size(); i++)
    LOG(INFO) << i << ": " << blob_names_[i];

  LOG(INFO) << "==============================================";
}

// DEPRECATED. As CPG_back function will not change the diff of param now.
template <typename Dtype>
void CPGLayer<Dtype>::Save_param_diff() {
  if (this->phase_ == TEST) return;

  //-----------------------------------------------------------------------
  // Save param
  //-----------------------------------------------------------------------
  /*const vector<shared_ptr<Blob<Dtype> > > params=net_->params();*/
  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  if (!is_history_init_) {
    history_params_.clear();

    /*const vector<string> param_display_names=net_->param_display_names();*/
    /*LOG(INFO) << "----------------------------------------------";*/
    /*LOG(INFO) << "The params to be saved are:";*/
    /*for (int param_id = 0; param_id < params.size(); ++param_id){*/
    /*history_params_.push_back(shared_ptr<Blob<Dtype> >(new
     * Blob<Dtype>(1,1,1,1)));*/
    /*LOG(INFO) << param_id<<": "<<param_display_names[param_id];*/
    /*}*/
    /*LOG(INFO) << "----------------------------------------------";*/

    const vector<string> layer_names = net_->layer_names();
    LOG(INFO) << "----------------------------------------------";
    LOG(INFO) << "The layer params to be saved are:";
    for (int layer_id = start_layer_index_; layer_id <= end_layer_index_;
         ++layer_id) {
      LOG(INFO) << "layer_id: " << layer_id
                << " layer_names: " << layer_names[layer_id];
      const int num_blobs = layers_[layer_id]->blobs().size();
      for (int blob_id = 0; blob_id < num_blobs; ++blob_id) {
        history_params_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(1, 1, 1, 1)));
        LOG(INFO) << "blob_id: " << blob_id;
      }
    }
    LOG(INFO) << "----------------------------------------------";
  }

  /*for (int param_id = 0; param_id < params.size(); ++param_id){*/
  /*history_params_[param_id]->CopyFrom(*params[param_id],true,true);*/
  /*history_params_[param_id]->CopyFrom(*params[param_id],false,true);*/
  /*}*/

  int p = 0;
  for (int layer_id = start_layer_index_; layer_id <= end_layer_index_;
       ++layer_id) {
    vector<shared_ptr<Blob<Dtype> > > layer_blobs = layers_[layer_id]->blobs();
    const int num_blobs = layer_blobs.size();
    for (int blob_id = 0; blob_id < num_blobs; ++blob_id) {
      history_params_[p]->CopyFrom(*layer_blobs[blob_id], true, true);
      history_params_[p]->CopyFrom(*layer_blobs[blob_id], false, true);
      p++;
    }
  }

  /*is_history_init_=true;*/
  /*return;*/
  //-----------------------------------------------------------------------
  // Save intermediate results
  //-----------------------------------------------------------------------
  const vector<shared_ptr<Blob<Dtype> > > blobs = net_->blobs();
  /*const vector<int> start_bottom_ids=net_->bottom_ids(start_index_);*/
  /*const vector<int> end_top_ids=net_->top_ids(end_index_);*/

  if (!is_history_init_) {
    history_blobs_.clear();
    const vector<string> blob_names = net_->blob_names();
    LOG(INFO) << "----------------------------------------------";
    LOG(INFO) << "The intermediate result blobs to be saved are:";
    /*for(int blob_id = start_bottom_ids[0]; blob_id <= end_top_ids[0];
     * ++blob_id)*/
    for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
      history_blobs_.push_back(
          shared_ptr<Blob<Dtype> >(new Blob<Dtype>(1, 1, 1, 1)));
      LOG(INFO) << "blob_id: " << blob_id
                << " blob_names: " << blob_names[blob_id];
    }
    LOG(INFO) << "----------------------------------------------";
  }

  /*for(int blob_id = start_bottom_ids[0]; blob_id <= end_top_ids[0];
   * ++blob_id)*/
  for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
    history_blobs_[blob_id]->CopyFrom(*blobs[blob_id], true, true);
    history_blobs_[blob_id]->CopyFrom(*blobs[blob_id], false, true);
  }

  is_history_init_ = true;
}

// DEPRECATED. As CPG_back function will not change the diff of param now.
template <typename Dtype>
void CPGLayer<Dtype>::Restore_param_diff() {
  if (this->phase_ == TEST) return;

  //-----------------------------------------------------------------------
  // Restore param
  //-----------------------------------------------------------------------
  /*const vector<shared_ptr<Blob<Dtype> > > params=net_->params();*/
  /*for (int param_id = 0; param_id < params.size(); ++param_id){*/
  /*[>LOG(INFO) <<"restore param_id: " <<param_id<<"
   * "<<params[param_id]->shape_string()<<"
   * "<<history_params_[param_id]->shape_string();<]*/
  /*params[param_id]->CopyFrom(*history_params_[param_id],true,false);*/
  /*[>LOG(INFO) <<"restore param_id: " <<param_id<<"
   * "<<params[param_id]->shape_string()<<"
   * "<<history_params_[param_id]->shape_string();<]*/
  /*params[param_id]->CopyFrom(*history_params_[param_id],false,false);*/
  /*}*/

  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  int p = 0;
  for (int layer_id = start_layer_index_; layer_id <= end_layer_index_;
       ++layer_id) {
    vector<shared_ptr<Blob<Dtype> > > layer_blobs = layers_[layer_id]->blobs();
    const int num_blobs = layer_blobs.size();
    for (int blob_id = 0; blob_id < num_blobs; ++blob_id) {
      layer_blobs[blob_id]->CopyFrom(*history_params_[p], true, false);
      layer_blobs[blob_id]->CopyFrom(*history_params_[p], false, false);
      p++;
    }
  }

  /*return;*/
  //-----------------------------------------------------------------------
  // Restore intermediate results
  //-----------------------------------------------------------------------
  const vector<shared_ptr<Blob<Dtype> > > blobs = net_->blobs();
  /*const vector<int> start_bottom_ids=net_->bottom_ids(start_index_);*/
  /*const vector<int> end_top_ids=net_->top_ids(end_index_);*/

  /*for(int blob_id = start_bottom_ids[0]; blob_id <= end_top_ids[0];
   * ++blob_id)*/
  for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
    /*LOG(INFO) <<"restore blob_id: " <<blob_id<<"
     * "<<blobs[blob_id]->shape_string()<<"
     * "<<history_blobs_[blob_id]->shape_string();*/
    blobs[blob_id]->CopyFrom(*history_blobs_[blob_id], true, false);
    blobs[blob_id]->CopyFrom(*history_blobs_[blob_id], false, false);
  }

  /*net_->ClearParamDiffs();*/
}

template <typename Dtype>
void Show_blob(const Dtype *data, const int channels, const int height,
               const int width, const string save_path,
               const string save_path_jet, const float threshold_ratio,
               const bool radioactive = false, const int fill = 0) {
  int rec_size = max(height, width) * 0.01;
  Dtype maxval = caffe_cpu_max_element(channels * height * width, data);
  Dtype sum = caffe_cpu_sum(channels * height * width, data);
  Dtype mean = sum / channels / height / width;

  Dtype threshold_value;
  if (threshold_ratio > 0) {
    threshold_value = maxval * threshold_ratio;
  } else {
    threshold_value = sum / channels / height / width;
  }
  Dtype scale_factor = 255.0 / threshold_value;

  //-----------------------------------------------------------------------
  cv::Mat cpg_mat;
  cv::Mat cpg_mat_jet;
  if (channels == 3) {
    cpg_mat = cv::Mat(height, width, CV_8UC3);
  } else if (channels == 1) {
    cpg_mat = cv::Mat(height, width, CV_8UC1);
  } else {
    LOG(FATAL) << "channels should 1 or 3";
  }

  uchar *cpg_mat_data = cpg_mat.data;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        int index_mat = (h * width + w) * channels + c;
        Dtype value = abs(data[index]);
        // Dtype value = data[index] > 0 ? data[index] : 0;
        if (value >= threshold_value) {
          cpg_mat_data[index_mat] = 255;
          //-----------------------------------------------------------------------
          if (radioactive) {
            for (int cc = 0; cc < channels; cc++) {
              for (int hh = max(h - rec_size, 0);
                   hh < min(h + rec_size, height); ++hh) {
                for (int ww = max(w - rec_size, 0);
                     ww < min(w + rec_size, width); ++ww) {
                  int index_mat_r = (hh * width + ww) * channels + cc;
                  cpg_mat_data[index_mat_r] = 255;

                  // int index_r = (cc * height + hh) * width + ww;
                  // Dtype value_r = abs(data[index_r]);
                  // if (value_r > threshold_value) {
                  // for (int ccc = 0; ccc < channels; ccc++) {
                  // for (int hhh = min(h, hh); hhh <= max(h, hh); ++hhh) {
                  // for (int www = min(w, ww); www <= max(w, ww); ++www) {
                  // int index_mat_r =
                  //(hhh * width + www) * channels + ccc;
                  // cpg_mat_data[index_mat_r] = 255;
                  //}
                  //}
                  //}
                  //}
                }
              }
            }
          }
          //-----------------------------------------------------------------------
        } else {
          if (fill >= 0) {
            cpg_mat_data[index_mat] = fill;
          } else {
            cpg_mat_data[index_mat] = scale_factor * value;
          }
        }
      }
    }
  }

  cv::imwrite(save_path, cpg_mat);
  LOG(INFO) << "radioactive: " << radioactive
            << " threshold_ratio: " << threshold_ratio
            << " threshold_value: " << threshold_value << " maxval: " << maxval
            << " mean: " << mean;
  LOG(INFO) << "save_path: " << save_path;

  cv::applyColorMap(cpg_mat, cpg_mat_jet, cv::COLORMAP_JET);
  cv::imwrite(save_path_jet, cpg_mat_jet);

  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // show the distubution of cpg_blob data
  // int total[26];
  // for (int i = 0; i < 26; ++i) {
  // total[i] = 0;
  //}
  // for (int e = 0; e < channels * height * width; e++) {
  // int level = int(data[e] / 10);
  // total[level]++;
  //}
  // for (int i = 0; i < 26; ++i) {
  // std::cout << i << ":" << total[i] << " ";
  //}
  // std::cout << std::endl;
  ////-----------------------------------------------------------------------
}

template <typename Dtype>
void CPGLayer<Dtype>::Show_im(const Dtype *im_data, const int current_label) {
  LOG(INFO) << "label: " << current_label;
  stringstream save_dir;
  save_dir << "tmp/" << voc_label_[current_label] << "/" << save_id_ << "/";
  boost::filesystem::create_directories(save_dir.str());

  stringstream save_path;
  save_path << save_dir.str() << "img.png";

  int channels= channels_cpg_;
  int height= height_im_;
  int width= width_im_;

  cv::Mat cpg_mat;
  if (channels == 3) {
    cpg_mat = cv::Mat(height, width, CV_8UC3);
  } else if (channels == 1) {
    cpg_mat = cv::Mat(height, width, CV_8UC1);
  } else {
    LOG(FATAL) << "channels should 1 or 3";
  }

  uchar *cpg_mat_data = cpg_mat.data;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        int index_mat = (h * width + w) * channels + c;
        Dtype value = im_data[index]+255.0/2.0;

          cpg_mat_data[index_mat] = value;

      }
    }
  }

  cv::imwrite(save_path.str(), cpg_mat);
  LOG(INFO) << "save_path: " << save_path.str();
}

template <typename Dtype>
void CPGLayer<Dtype>::Show_cpg(const Dtype *cpg_data, const int current_label,
                               const string info) {
  // 除了原始CPG外，高于阈值的像素点用255表示，低于阈值的像素点用0表示
  LOG(INFO) << "label: " << current_label << " info: " << info;
  // string save_subdir = currentDateTime();
  stringstream save_dir;
  save_dir << "tmp/" << voc_label_[current_label] << "/" << save_id_ << "/";
  boost::filesystem::create_directories(save_dir.str());

  stringstream save_path;
  stringstream save_path_jet;
  save_path << save_dir.str() << "_o" << info << ".png";
  save_path_jet << save_dir.str() << "jet_o" << info << ".png";
  Show_blob(cpg_data, channels_cpg_, height_im_, width_im_, save_path.str(),
            save_path_jet.str(), 1, false, -1);

  save_path.str(std::string());
  save_path_jet.str(std::string());
  save_path << save_dir.str() << "_0" << info << ".png";
  save_path_jet << save_dir.str() << "jet_0" << info << ".png";
  Show_blob(cpg_data, channels_cpg_, height_im_, width_im_, save_path.str(),
            save_path_jet.str(), 0, false);

  for (int t = 1; t < 4; ++t) {
    save_path.str(std::string());
    save_path_jet.str(std::string());
    save_path << save_dir.str() << "_" << pow(10, -t) << info << ".png";
    save_path_jet << save_dir.str() << "jet_" << pow(10, -t) << info << ".png";
    Show_blob(cpg_data, channels_cpg_, height_im_, width_im_, save_path.str(),
              save_path_jet.str(), pow(10, -t), false);

    save_path.str(std::string());
    save_path_jet.str(std::string());
    save_path << save_dir.str() << "_" << pow(10, -t) << "_ra" << info
              << ".png";
    save_path_jet << save_dir.str() << "jet_" << pow(10, -t) << "_ra" << info
                  << ".png";
    Show_blob(cpg_data, channels_cpg_, height_im_, width_im_, save_path.str(),
              save_path_jet.str(), pow(10, -t), true);
  }
}

template <typename Dtype>
void CPGLayer<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<vector<Blob<Dtype> *> > bottom_vecs_ = net_->bottom_vecs();
  const vector<string> blob_names_ = net_->blob_names();
  const vector<string> layer_names_ = net_->layer_names();
  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  const vector<vector<bool> > bottom_need_backward_ =
      net_->bottom_need_backward();

  const vector<Blob<Dtype> *> &bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    /*if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }*/
    const Blob<Dtype> &blob = *bottom_vec[bottom_id];

    const vector<int> tmp = net_->bottom_ids(layer_id);
    const string &blob_name = blob_names_[tmp[bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Backward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", bottom blob " << blob_name
                                       << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    /*if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }*/
    const Blob<Dtype> &blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Backward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", param blob " << param_id
                                       << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void CPGLayer<Dtype>::Clear_split_diff() {
  for (size_t i = 0; i < split_top_blob_.size(); ++i) {
    caffe_gpu_set(split_top_blob_[i]->count(), static_cast<Dtype>(0),
                  split_top_blob_[i]->mutable_gpu_diff());
  }
}

template <typename Dtype>
bool CPGLayer<Dtype>::Need_Repartition(const int cls_id, const Dtype label,
                                       const Dtype predict) {
  if (cls_id == ignore_label_) return false;
  // assume label is betwween 0 ~ 1
  if (this->phase_ == TRAIN) {
    if (label < 0.5) return false;
    if (predict >= predict_threshold_) {
      return true;
    } else {
      return false;
    }
  } else if (this->phase_ == TEST) {
    // label is equal to predict in test
    if (label >= predict_threshold_) {
      return true;
    } else {
      return false;
    }
  } else {
    LOG(FATAL) << "unkown phase: " << this->phase_;
  }
  LOG(FATAL) << "We should not arrive here!";
  return false;
}

template <typename Dtype>
bool CPGLayer<Dtype>::Need_Order(const int cls_id, const Dtype label,
                                 const Dtype predict) {
  if (cls_id == ignore_label_) return false;
  // assum score is betwween 0 ~ 1
  if (this->phase_ == TRAIN) {
    if (label < 0.5) return false;
    if (is_order_ && predict > predict_order_) {
      return true;
    } else {
      return false;
    }
  } else if (this->phase_ == TEST) {
    return false;
  } else {
    LOG(FATAL) << "unkown phase: " << this->phase_;
  }
  LOG(FATAL) << "We should not arrive here!";
  return false;
}

template <typename Dtype>
void CPGLayer<Dtype>::CPG_back() {
  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  const vector<vector<Blob<Dtype> *> > top_vecs_ = net_->top_vecs();
  const vector<vector<Blob<Dtype> *> > bottom_vecs_ = net_->bottom_vecs();

  if (bottom_need_backward_.size() == 0) {
    bottom_need_backward_ = net_->bottom_need_backward();
    for (int layer_id = end_layer_index_; layer_id >= start_layer_index_;
         --layer_id) {
      std::fill(bottom_need_backward_[layer_id].begin(),
                bottom_need_backward_[layer_id].end(), true);

      vector<bool> this_param_propagate_down;
      for (int j = 0; j < layers_[layer_id]->blobs().size(); j++) {
        this_param_propagate_down.push_back(
            layers_[layer_id]->param_propagate_down(j));
      }
      origin_param_propagate_down_.push_back(this_param_propagate_down);
    }
  }

  for (int layer_id = end_layer_index_; layer_id >= start_layer_index_;
       --layer_id) {
    for (int j = 0; j < layers_[layer_id]->blobs().size(); j++) {
      layers_[layer_id]->set_param_propagate_down(j, false);
    }

    layers_[layer_id]->Backward(top_vecs_[layer_id],
                                bottom_need_backward_[layer_id],
                                bottom_vecs_[layer_id]);

    for (int j = 0; j < layers_[layer_id]->blobs().size(); j++) {
      layers_[layer_id]->set_param_propagate_down(
          j, origin_param_propagate_down_[end_layer_index_ - layer_id][j]);
    }

    if (debug_info_) {
      BackwardDebugInfo(layer_id);
    }
  }
}

template <typename Dtype>
__global__ void filter_kernel(const int count, const Dtype *const in,
                              const int height, const int width,
                              Dtype *const out) {
  CUDA_KERNEL_LOOP(index, count) {
    const int h = index / width;
    const int w = index % width;

    const int offset = 1;
    const int hstart = max(h - offset, 0);
    const int hend = min(h + offset, height - 1);
    const int wstart = max(w - offset, 0);
    const int wend = min(w + offset, width - 1);

    Dtype sum = 0;
    Dtype scale = 0;
    for (int i = hstart; i <= hend; ++i) {
      for (int j = wstart; j <= wend; ++j) {
        sum += in[i * width + j];
        scale++;
      }
      out[index] = sum / scale;
    }
  }
}

template <typename Dtype>
__global__ void Do_threshold(const int count, Dtype *const data,
                             const Dtype thr, const Dtype rep) {
  CUDA_KERNEL_LOOP(index, count) {
    if (data[index] < thr) data[index] = rep;
  }
}

template <typename Dtype>
__global__ void tanh_gpu(const int count, Dtype *const data) {
  CUDA_KERNEL_LOOP(index, count) { data[index] = tanh(data[index]); }
}

template <typename Dtype>
void CPGLayer<Dtype>::After() {
  pass_im_ += num_im_;


  accum_im_ += num_im_;
  accum_gt_ += gt_class_.size();
  accum_bp_ += bp_class_.size();
  if (pass_im_ % display_ == 0) {
    LOG(INFO) << "is_cpg: " << is_cpg_ << " #im: " << accum_im_
              << " #bp: " << accum_bp_ << " #gt: " << accum_gt_;
    accum_im_ = 0;
    accum_gt_ = 0;
    accum_bp_ = 0;
  }
  save_id_ += num_im_;
}

template <typename Dtype>
void CPGLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  if (max_num_im_cpg_ >= 0 && max_num_im_cpg_ <= pass_im_) {
    is_cpg_ = false;
  }
  if (!is_cpg_) {
    return;
  }

  //-----------------------------------------------------------------------
  // Show info
  //-----------------------------------------------------------------------
  Show_info();
  LOG_IF(INFO, debug_info_) << "------------------start-----------------------";
  LOG_IF(INFO, debug_info_) << "save_id_: " << save_id_;

  //-----------------------------------------------------------------------
  // save the history param diff
  //-----------------------------------------------------------------------
  /*Save_param_diff();*/

  //-----------------------------------------------------------------------
  // clear split layer diff
  // When CPG backward, the split layer will sum up the unrelate diff of other
  // stream, so set all diff in split layer to zero.
  Clear_split_diff();

  const Dtype *bottom_label = bottom[bottom_label_index_]->cpu_data();
  const Dtype *predict_data = predict_blob_->cpu_data();

  bp_class_.clear();
  gt_class_.clear();
  for (int cls_id = 0; cls_id < num_class_; ++cls_id) {
    int index = cls_id;
    LOG_IF(INFO, debug_info_) << "class: " << voc_label_[cls_id]
                              << "\t\tlabel: " << bottom_label[index]
                              << " score: " << predict_data[index];
    if (Need_Repartition(cls_id, bottom_label[index], predict_data[index])) {
    } else if (Need_Order(cls_id, bottom_label[index], predict_data[index])) {
    } else {
      continue;
    }
    bp_class_.push_back(cls_id);
    gt_class_.push_back(cls_id);
    LOG_IF(INFO, debug_info_) << "gt class: " << voc_label_[cls_id];
  }
  if (gt_class_.size() == 0) {
    After();
    LOG_IF(INFO, debug_info_) << "Nothing to BP ";
    return;
  }

  if (is_contrast_) {
    while (bp_class_.size() < 3) {
      Dtype max_score = -FLT_MAX;
      int max_id = -1;
      for (int i = 0; i < num_class_; ++i) {
        if (std::find(bp_class_.begin(), bp_class_.end(), i) !=
            bp_class_.end()) {
          /* v contains x */
          continue;
        } else {
          /* v does not contain x */
          int index = i;
          if (predict_data[index] > max_score) {
            max_id = i;
            max_score = predict_data[index];
          }
        }
      }
      CHECK_NE(max_id, -1) << "max_id can not be -1";
      bp_class_.push_back(max_id);
      LOG_IF(INFO, debug_info_) << "addition class: " << voc_label_[max_id];
    }
  }

  // shape the blob to save cpg_blob
  raw_cpg_->Reshape(bp_class_.size(), cpg_blob_.size(), height_im_, width_im_);
  caffe_gpu_set(raw_cpg_->count(), Dtype(0), raw_cpg_->mutable_gpu_data());
  caffe_gpu_set(raw_cpg_->count(), Dtype(0), raw_cpg_->mutable_gpu_diff());

  //-----------------------------------------------------------------------
  // me back
  //-----------------------------------------------------------------------
  for (size_t bp_id = 0; bp_id < bp_class_.size(); ++bp_id) {
    int index = bp_class_[bp_id];
    LOG_IF(INFO, debug_info_) << "class: " << voc_label_[bp_class_[bp_id]]
                              << " label: " << bottom_label[index]
                              << " score: " << predict_data[index];

    caffe_gpu_set(predict_blob_->count(), static_cast<Dtype>(0),
                  predict_blob_->mutable_gpu_diff());
    caffe_gpu_set(im_blob_->count(), static_cast<Dtype>(0),
                  im_blob_->mutable_gpu_diff());

    Dtype *predict_diff = predict_blob_->mutable_cpu_diff();
    predict_diff[index] = predict_data[index];

    // Gradient 1
    // predict_diff[index] = 1;

    // Gradient all
    // caffe_copy(predict_blob_->count(), predict_blob_->gpu_data(),
    // predict_blob_->mutable_gpu_diff());

    if (predict_data[index] == 1.0) {
      predict_diff[index] = 0.99999;
    } else {
    }

    // if test, we always try find cache first
    if (this->phase_ == TEST && false) {
      bool is_back = false;
      BlobProto cache_proto;
      Blob<Dtype> cache_blob;
      stringstream cache_path;
      for (size_t blob_id = 0; blob_id < cpg_blob_.size(); ++blob_id) {
        cache_path.str(string());
        cache_path << "data/cpg_cache_test/" << save_id_ << "_"
                   << bp_class_[bp_id] << "_" << cpg_blob_name_[blob_id];

        if (boost::filesystem::exists(cache_path.str())) {
          ReadProtoFromBinaryFileOrDie(cache_path.str(), &cache_proto);
          cache_blob.FromProto(cache_proto, true);
          caffe_copy(cpg_blob_[blob_id]->count(), cache_blob.gpu_data(),
                     cpg_blob_[blob_id]->mutable_gpu_diff());
        } else {
          is_back = true;
          break;
        }
      }

      if (is_back) {
        CPG_back();

        boost::filesystem::create_directories("data/cpg_cache_test");
        for (size_t blob_id = 0; blob_id < cpg_blob_.size(); ++blob_id) {
          cache_path.str(string());
          cache_path << "data/cpg_cache_test/" << save_id_ << "_"
                     << bp_class_[bp_id] << "_" << cpg_blob_name_[blob_id];

          cache_blob.ReshapeLike(*cpg_blob_[blob_id]);
          caffe_copy(cpg_blob_[blob_id]->count(),
                     cpg_blob_[blob_id]->gpu_diff(),
                     cache_blob.mutable_gpu_data());
          cache_blob.ToProto(&cache_proto, false);
          WriteProtoToBinaryFile(cache_proto, cache_path.str());
        }
      }
    } else {
      CPG_back();
    }

    // maximum along channel and save to raw_cpg_ diff
    for (size_t blob_id = 0; blob_id < cpg_blob_.size(); ++blob_id) {
      const int channels_cpg_this = cpg_blob_[blob_id]->shape(1);
      const int size_cpg_this =
          cpg_blob_[blob_id]->shape(2) * cpg_blob_[blob_id]->shape(3);

      caffe_gpu_abs(cpg_blob_[blob_id]->count(), cpg_blob_[blob_id]->gpu_diff(),
                    cpg_blob_[blob_id]->mutable_gpu_diff());

      if (channels_cpg_this == 1) {
        caffe_copy(size_cpg_this, cpg_blob_[blob_id]->gpu_diff(),
                   raw_cpg_->mutable_gpu_diff() +
                       raw_cpg_->offset(bp_id, blob_id, 0, 0));
      } else {
        caffe_gpu_maximum(size_cpg_this,
                          cpg_blob_[blob_id]->gpu_diff() + size_cpg_this * 0,
                          cpg_blob_[blob_id]->gpu_diff() + size_cpg_this * 1,
                          raw_cpg_->mutable_gpu_diff() +
                              raw_cpg_->offset(bp_id, blob_id, 0, 0));

        for (int i = 2; i < channels_cpg_this; ++i) {
          caffe_gpu_maximum(
              size_cpg_this, cpg_blob_[blob_id]->gpu_diff() + size_cpg_this * i,
              raw_cpg_->gpu_diff() + raw_cpg_->offset(bp_id, blob_id, 0, 0),
              raw_cpg_->mutable_gpu_diff() +
                  raw_cpg_->offset(bp_id, blob_id, 0, 0));
        }
      }
    }
  }

  if (is_contrast_) {
    caffe_copy(raw_cpg_->count(), raw_cpg_->gpu_diff(),
               raw_cpg_->mutable_gpu_data());
    for (size_t gt_id = 0; gt_id < gt_class_.size(); ++gt_id) {
      for (size_t bp_id = 0; bp_id < bp_class_.size(); ++bp_id) {
        if (gt_id == bp_id) continue;
        for (size_t blob_id = 0; blob_id < cpg_blob_.size(); ++blob_id) {
          const int channel_size =
              cpg_blob_[blob_id]->shape(2) * cpg_blob_[blob_id]->shape(3);

          caffe_gpu_sub(
              channel_size,
              raw_cpg_->gpu_data() + raw_cpg_->offset(gt_id, blob_id, 0, 0),
              raw_cpg_->gpu_data() + raw_cpg_->offset(bp_id, blob_id, 0, 0),
              raw_cpg_->mutable_gpu_diff() +
                  raw_cpg_->offset(bp_id, blob_id, 0, 0));

          // NOLINT_NEXT_LINE(whitespace/operators)
          Do_threshold<Dtype><<<CAFFE_GET_BLOCKS(channel_size),
                                CAFFE_CUDA_NUM_THREADS>>>(
              channel_size, raw_cpg_->mutable_gpu_diff() +
                                raw_cpg_->offset(bp_id, blob_id, 0, 0),
              0, 0);

          caffe_gpu_add(
              channel_size,
              raw_cpg_->gpu_diff() + raw_cpg_->offset(gt_id, blob_id, 0, 0),
              raw_cpg_->gpu_diff() + raw_cpg_->offset(bp_id, blob_id, 0, 0),
              raw_cpg_->mutable_gpu_diff() +
                  raw_cpg_->offset(gt_id, blob_id, 0, 0));
        }
      }
    }
  }

  for (size_t gt_id = 0; gt_id < gt_class_.size(); ++gt_id) {
    int cls_id = gt_class_[gt_id];
    if (cpg_blob_.size() == 1 && cpg_blob_[0]->shape(2) == height_im_ &&
        cpg_blob_[0]->shape(3) == width_im_) {
      caffe_copy(size_cpg_,
                 raw_cpg_->gpu_diff() + raw_cpg_->offset(gt_id, 0, 0, 0),
                 top[0]->mutable_gpu_data() + top[0]->offset(0, cls_id, 0, 0));
    } else {
      // resize to image size and average
      // from cpg_blob_ diff to data
      for (size_t blob_id = 0; blob_id < cpg_blob_.size(); ++blob_id) {
        if (cpg_blob_[blob_id]->shape(2) == height_im_ &&
            cpg_blob_[blob_id]->shape(3) == width_im_) {
          caffe_copy(size_cpg_, raw_cpg_->gpu_diff() +
                                    raw_cpg_->offset(gt_id, blob_id, 0, 0),
                     raw_cpg_->mutable_gpu_data() +
                         raw_cpg_->offset(gt_id, blob_id, 0, 0));
        } else {
          caffe_gpu_interp2<Dtype, false>(
              1, raw_cpg_->gpu_diff() + raw_cpg_->offset(gt_id, blob_id, 0, 0),
              0, 0, cpg_blob_[blob_id]->shape(2), cpg_blob_[blob_id]->shape(3),
              cpg_blob_[blob_id]->shape(2), cpg_blob_[blob_id]->shape(3),
              raw_cpg_->mutable_gpu_data() +
                  raw_cpg_->offset(gt_id, blob_id, 0, 0),
              0, 0, height_im_, width_im_, height_im_, width_im_);
        }

        int max_value_index;
        caffe_gpu_amax(size_cpg_, raw_cpg_->gpu_data() +
                                      raw_cpg_->offset(gt_id, blob_id, 0, 0),
                       &max_value_index);
        max_value_index--;
        const Dtype maxval =
            *(raw_cpg_->cpu_data() + raw_cpg_->offset(gt_id, blob_id, 0, 0) +
              max_value_index);

        caffe_gpu_scal(size_cpg_, Dtype(1.0 / maxval),
                       raw_cpg_->mutable_gpu_data() +
                           raw_cpg_->offset(gt_id, blob_id, 0, 0));

        caffe_gpu_add(
            size_cpg_, top[0]->gpu_data() + top[0]->offset(0, cls_id, 0, 0),
            raw_cpg_->gpu_data() + raw_cpg_->offset(gt_id, blob_id, 0, 0),
            top[0]->mutable_gpu_data() + top[0]->offset(0, cls_id, 0, 0));
        if (debug_info_) {
          Show_cpg(
              raw_cpg_->cpu_data() + raw_cpg_->offset(gt_id, blob_id, 0, 0),
              cls_id, "_" + cpg_blob_name_[blob_id]);
        }
      }
    }

    // 平滑滤波
    if (false) {
      const int top_offset = top[0]->height() * top[0]->width();
      // NOLINT_NEXT_LINE(whitespace/operators)
      for (int i = 0; i < top[0]->num() * top[0]->channels(); ++i) {
        filter_kernel<
            Dtype><<<CAFFE_GET_BLOCKS(top_offset), CAFFE_CUDA_NUM_THREADS>>>(
            top_offset, top[0]->gpu_data() + i * top_offset, top[0]->height(),
            top[0]->width(), top[0]->mutable_gpu_diff() + i * top_offset);
      }
      caffe_copy(top[0]->count(), top[0]->gpu_diff(),
                 top[0]->mutable_gpu_data());
    }

    if (debug_info_) {
      Show_im(im_blob_->cpu_data(), cls_id);
      Show_cpg(top[0]->cpu_data() + top[0]->offset(0, cls_id, 0, 0), cls_id,
               "_fusion");
    }

    //-----------------------------------------------------------------------
  }

  //-----------------------------------------------------------------------
  // restore history param diff
  //-----------------------------------------------------------------------
  // Restore_param_diff();

  After();
}

template <typename Dtype>
void CPGLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                   const vector<bool> &propagate_down,
                                   const vector<Blob<Dtype> *> &bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CPGLayer);

}  // namespace caffe
