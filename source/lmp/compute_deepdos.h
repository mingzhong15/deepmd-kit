// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(deepdos, ComputeDeepdos)
// clang-format on
#else

#ifndef LMP_COMPUTE_DEEPDOS_H
#define LMP_COMPUTE_DEEPDOS_H

#include "compute.h"
#include "pair_deepmd.h"
#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepDOS.h"
#else
#include "deepmd/DeepDOS.h"
#endif
namespace deepmd_compat = deepmd;
#else
// DeepDOS is only available with DP_USE_CXX_API.
// The constructor will issue an error at runtime.
namespace deepmd_compat {
class DeepDOS {
 public:
  void init(const std::string&, const int&) {}
  int numb_dos() const { return 0; }
  double cutoff() const { return 0.0; }
};
}  // namespace deepmd_compat
#endif

namespace LAMMPS_NS {

class ComputeDeepdos : public Compute {
 public:
  ComputeDeepdos(class LAMMPS*, int, char**);
  ~ComputeDeepdos() override;
  void init() override;
  void compute_peratom() override;
  void compute_vector() override;
  double memory_usage() override;
  void init_list(int, class NeighList*) override;
  double dist_unit_cvt_factor;

 private:
  enum DosMode { ATOM, TOTAL };

  int nmax;
  DosMode mode;
  double** dos_atom_array;
  double* dos_vector;
  PairDeepMD dp;
  class NeighList* list;
  deepmd_compat::DeepDOS dd;
  int numb_dos;
};

}  // namespace LAMMPS_NS

#endif
#endif
