// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepDOS.h"

#include <memory>

#include "common.h"

#ifdef BUILD_PYTORCH
#include "DeepDOSPT.h"
#endif

using namespace deepmd;

DeepDOS::DeepDOS() : inited(false) {}

DeepDOS::DeepDOS(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content)
    : inited(false) {
  init(model, gpu_rank, file_content);
}

DeepDOS::~DeepDOS() {}

void DeepDOS::init(const std::string& model,
                   const int& gpu_rank,
                   const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  const DPBackend backend = get_backend(model);
  if (deepmd::DPBackend::PyTorch == backend) {
#ifdef BUILD_PYTORCH
    dos = std::make_shared<deepmd::DeepDOSPT>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception("PyTorch backend is not built.");
#endif
  } else {
    throw deepmd::deepmd_exception("Unknown file type");
  }
  inited = true;
}

void DeepDOS::print_summary(const std::string& pre) const {
  deepmd::print_summary(pre);
}

template <typename VALUETYPE>
void DeepDOS::compute(std::vector<VALUETYPE>& dos,
                      std::vector<VALUETYPE>& atom_dos,
                      const std::vector<VALUETYPE>& coord,
                      const std::vector<int>& atype,
                      const std::vector<VALUETYPE>& box,
                      const int nghost,
                      const InputNlist& inlist) {
  this->dos->computew(dos, atom_dos, coord, atype, box, nghost, inlist);
}

template void DeepDOS::compute<double>(std::vector<double>& dos,
                                       std::vector<double>& atom_dos,
                                       const std::vector<double>& coord,
                                       const std::vector<int>& atype,
                                       const std::vector<double>& box,
                                       const int nghost,
                                       const InputNlist& inlist);

template void DeepDOS::compute<float>(std::vector<float>& dos,
                                      std::vector<float>& atom_dos,
                                      const std::vector<float>& coord,
                                      const std::vector<int>& atype,
                                      const std::vector<float>& box,
                                      const int nghost,
                                      const InputNlist& inlist);

double DeepDOS::cutoff() const { return dos->cutoff(); }

int DeepDOS::numb_types() const { return dos->numb_types(); }

int DeepDOS::numb_dos() const { return dos->numb_dos(); }

void DeepDOS::get_type_map(std::string& type_map) {
  dos->get_type_map(type_map);
}
