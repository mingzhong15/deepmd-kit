// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "DeepDOSPT.h"

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/runtime/jit_exception.h>

#include <cstdint>

#include "common.h"
#include "commonPT.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

void DeepDOSPT::translate_error(std::function<void()> f) {
  try {
    f();
  } catch (const c10::Error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  } catch (const torch::jit::JITException& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend JIT error: " +
                                   std::string(e.what()));
  } catch (const std::runtime_error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  }
}

DeepDOSPT::DeepDOSPT() : inited(false) {}

DeepDOSPT::DeepDOSPT(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false) {
  try {
    translate_error([&] { init(model, gpu_rank, file_content); });
  } catch (...) {
    throw;
  }
}

void DeepDOSPT::init(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  deepmd::load_op_library();
  int gpu_num = torch::cuda::device_count();
  gpu_id = (gpu_num > 0) ? (gpu_rank % gpu_num) : 0;
  gpu_enabled = torch::cuda::is_available();
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
    std::cout << "load model from: " << model << " to cpu " << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  std::unordered_map<std::string, std::string> metadata = {{"type", ""}};
  module = torch::jit::load(model, device, metadata);
  module.eval();
  do_message_passing = module.run_method("has_message_passing").toBool();

  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  if (num_inter_nthreads) {
    try {
      at::set_num_interop_threads(num_inter_nthreads);
    } catch (...) {
    }
  }
  if (num_intra_nthreads) {
    try {
      at::set_num_threads(num_intra_nthreads);
    } catch (...) {
    }
  }

  auto rcut_ = module.run_method("get_rcut").toDouble();
  rcut = static_cast<double>(rcut_);
  ntypes = module.run_method("get_ntypes").toInt();
  ndos = module.run_method("get_numb_dos").toInt();
  inited = true;
}

DeepDOSPT::~DeepDOSPT() {}

void DeepDOSPT::get_type_map(std::string& type_map) {
  auto ret = module.run_method("get_type_map").toList();
  type_map.clear();
  for (const torch::IValue& element : ret) {
    if (!type_map.empty()) {
      type_map += " ";
    }
    type_map += torch::str(element);
  }
}

template <typename VALUETYPE>
void DeepDOSPT::compute(std::vector<VALUETYPE>& dos,
                        std::vector<VALUETYPE>& atom_dos,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const int nghost,
                        const InputNlist& lmp_list) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  // select real atoms (filter out NULL-type virtual atoms)
  std::vector<VALUETYPE> dcoord, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  int nframes = 1;
  std::vector<VALUETYPE> aparam;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, nframes, 0, nall, false);
  // Detect whether any NULL-type atoms were filtered out.
  bool has_null_atoms = (nall_real < nall);

  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option).to(device);

  // copy and process neighbor list
  nlist_data.copy_from_nlist(lmp_list, nall - nghost);
  nlist_data.shuffle_exclude_empty(fwd_map);
  nlist_data.padding();
  if (do_message_passing) {
    if (!lmp_list.sendlist) {
      throw deepmd::deepmd_exception(
          "Message-passing model requires a full communication-aware "
          "neighbor list, which is not available in standalone compute "
          "mode. Use a pair style instead.");
    }
    if (has_null_atoms) {
      build_comm_dict_with_virtual_atoms(
          comm_dict, lmp_list, fwd_map, remapped_sendlist,
          remapped_sendlist_ptrs, remapped_sendnum, remapped_recvnum);
    } else {
      build_comm_dict(comm_dict, lmp_list, lmp_list.sendlist,
                      lmp_list.sendnum, lmp_list.recvnum);
    }
  }

  c10::optional<torch::Tensor> fparam_tensor;
  c10::optional<torch::Tensor> aparam_tensor;

  // mapping for attention-based models
  if (lmp_list.mapping) {
    std::vector<std::int64_t> mapping(nall_real);
    for (size_t ii = 0; ii < nall_real; ii++) {
      mapping[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
    }
    mapping_tensor =
        torch::from_blob(mapping.data(), {1, nall_real}, int_option)
            .to(device);
  } else {
    mapping_tensor = c10::optional<torch::Tensor>();
  }

  at::Tensor firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);

  bool do_atom_virial_tensor = false;

  // forward pass through the model
  c10::IValue outputs_ival;
  if (do_message_passing) {
    outputs_ival = module.run_method(
        "forward_lower", coord_wrapped_Tensor, atype_Tensor, firstneigh_tensor,
        mapping_tensor, fparam_tensor, aparam_tensor, do_atom_virial_tensor,
        comm_dict);
  } else {
    outputs_ival = module.run_method(
        "forward_lower", coord_wrapped_Tensor, atype_Tensor, firstneigh_tensor,
        mapping_tensor, fparam_tensor, aparam_tensor, do_atom_virial_tensor);
  }
  auto outputs = outputs_ival.toGenericDict();

  // extract global dos
  dos.clear();
  if (outputs.contains(c10::IValue("dos"))) {
    c10::IValue dos_ = outputs.at("dos");
    torch::Tensor flat_dos_ = dos_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_dos_ = flat_dos_.to(torch::kCPU);
    dos.assign(cpu_dos_.data_ptr<VALUETYPE>(),
               cpu_dos_.data_ptr<VALUETYPE>() + cpu_dos_.numel());
  }

  // extract atomic dos
  c10::IValue atom_dos_ = outputs.at(c10::IValue("atom_dos"));
  torch::Tensor flat_atom_dos_ = atom_dos_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_dos_ = flat_atom_dos_.to(torch::kCPU);
  std::vector<VALUETYPE> datom_dos;
  datom_dos.assign(cpu_atom_dos_.data_ptr<VALUETYPE>(),
                   cpu_atom_dos_.data_ptr<VALUETYPE>() +
                       cpu_atom_dos_.numel());
  // map atom_dos back to original atom ordering
  atom_dos.resize(static_cast<size_t>(nframes) * fwd_map.size() * ndos);
  select_map<VALUETYPE>(atom_dos, datom_dos, bkw_map, ndos, nframes,
                        fwd_map.size(), nall_real);
}

// Explicit template instantiations
template void DeepDOSPT::compute<double>(std::vector<double>& dos,
                                         std::vector<double>& atom_dos,
                                         const std::vector<double>& coord,
                                         const std::vector<int>& atype,
                                         const std::vector<double>& box,
                                         const int nghost,
                                         const InputNlist& inlist);
template void DeepDOSPT::compute<float>(std::vector<float>& dos,
                                        std::vector<float>& atom_dos,
                                        const std::vector<float>& coord,
                                        const std::vector<int>& atype,
                                        const std::vector<float>& box,
                                        const int nghost,
                                        const InputNlist& inlist);

// public wrapper methods
void DeepDOSPT::computew(std::vector<double>& dos,
                         std::vector<double>& atom_dos,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const int nghost,
                         const InputNlist& inlist) {
  translate_error([&] {
    compute(dos, atom_dos, coord, atype, box, nghost, inlist);
  });
}

void DeepDOSPT::computew(std::vector<float>& dos,
                         std::vector<float>& atom_dos,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const int nghost,
                         const InputNlist& inlist) {
  translate_error([&] {
    compute(dos, atom_dos, coord, atype, box, nghost, inlist);
  });
}
#endif
