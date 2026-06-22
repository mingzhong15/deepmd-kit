// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(deepmd/thermo, ComputeDeepmdThermo)
// clang-format on
#else

#ifndef LMP_COMPUTE_DEEPMD_THERMO_H
#define LMP_COMPUTE_DEEPMD_THERMO_H

#include "compute.h"
#include "pair_deepmd.h"

namespace LAMMPS_NS {

class ComputeDeepmdThermo : public Compute {
 public:
  ComputeDeepmdThermo(class LAMMPS*, int, char**);
  ~ComputeDeepmdThermo() override;
  void init() override;
  double compute_scalar() override;

 private:
  enum ThermoType { FREE_ENERGY, ELE_ENTROPY, INTERNAL_ENERGY };

  ThermoType thermo_type;
  PairDeepMD* pair;
};

}  // namespace LAMMPS_NS

#endif
#endif
