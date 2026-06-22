// SPDX-License-Identifier: LGPL-3.0-or-later
#include "compute_deepmd_thermo.h"

#include <cstring>

#include "error.h"
#include "force.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeDeepmdThermo::ComputeDeepmdThermo(LAMMPS* lmp, int narg, char** arg)
    : Compute(lmp, narg, arg), pair(nullptr) {
  if (narg < 4) {
    error->all(FLERR, "Illegal compute deepmd/thermo command");
  }

  std::string keyword = arg[3];
  if (keyword == "free_energy") {
    thermo_type = FREE_ENERGY;
  } else if (keyword == "ele_entropy") {
    thermo_type = ELE_ENTROPY;
  } else if (keyword == "internal_energy") {
    thermo_type = INTERNAL_ENERGY;
  } else {
    error->all(FLERR,
               "Illegal compute deepmd/thermo command: unknown keyword "
               "argument. Must be one of: free_energy, ele_entropy, "
               "internal_energy");
  }

  scalar_flag = 1;
  extscalar = 1;
  timeflag = 1;
}

/* ---------------------------------------------------------------------- */

ComputeDeepmdThermo::~ComputeDeepmdThermo() = default;

/* ---------------------------------------------------------------------- */

void ComputeDeepmdThermo::init() {
  if (!force->pair) {
    error->all(FLERR,
               "No pair style is defined for compute deepmd/thermo");
  }
  pair = dynamic_cast<PairDeepMD*>(force->pair);
  if (!pair) {
    error->all(FLERR,
               "compute deepmd/thermo requires a pair style of type deepmd");
  }
}

/* ---------------------------------------------------------------------- */

double ComputeDeepmdThermo::compute_scalar() {
  invoked_scalar = update->ntimestep;

  switch (thermo_type) {
    case FREE_ENERGY:
      return pair->get_free_energy_LAMMPS();
    case ELE_ENTROPY:
      return pair->get_ele_entropy_LAMMPS();
    case INTERNAL_ENERGY:
      return pair->get_internal_energy_LAMMPS();
  }
  return 0.0;
}
