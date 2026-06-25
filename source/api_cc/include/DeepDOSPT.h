// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include "DeepDOS.h"

namespace deepmd {

class DeepDOSPT : public DeepDOSBase {
 public:
  DeepDOSPT();
  virtual ~DeepDOSPT();
  DeepDOSPT(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

  double cutoff() const {
    assert(inited);
    return rcut;
  };
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  int numb_dos() const {
    assert(inited);
    return ndos;
  };
  void get_type_map(std::string& type_map);

  void computew(std::vector<double>& dos,
                std::vector<double>& atom_dos,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist);
  void computew(std::vector<float>& dos,
                std::vector<float>& atom_dos,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist);

 private:
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& dos,
               std::vector<VALUETYPE>& atom_dos,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist);

  int num_intra_nthreads, num_inter_nthreads;
  bool inited;
  double rcut;
  int ntypes;
  int ndos;
  int gpu_id;
  bool gpu_enabled;
  bool do_message_passing;
  torch::jit::script::Module module;
  NeighborListData nlist_data;
  at::Tensor firstneigh_tensor;
  c10::optional<torch::Tensor> mapping_tensor;
  torch::Dict<std::string, torch::Tensor> comm_dict;
  std::vector<std::vector<int>> remapped_sendlist;
  std::vector<int*> remapped_sendlist_ptrs;
  std::vector<int> remapped_sendnum;
  std::vector<int> remapped_recvnum;

  void translate_error(std::function<void()> f);
};

}  // namespace deepmd
