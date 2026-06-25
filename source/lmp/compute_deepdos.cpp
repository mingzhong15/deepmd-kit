// SPDX-License-Identifier: LGPL-3.0-or-later
#include "compute_deepdos.h"

#ifdef DP_USE_CXX_API

#include <cstring>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

using namespace LAMMPS_NS;

#define VALUETYPE double

/* ---------------------------------------------------------------------- */

ComputeDeepdos::ComputeDeepdos(LAMMPS* lmp, int narg, char** arg)
    : Compute(lmp, narg, arg),
      dp(lmp),
      dos_atom_array(nullptr),
      dos_vector(nullptr),
      list(nullptr) {
  if (strcmp(update->unit_style, "lj") == 0) {
    error->all(FLERR,
               "Compute deepdos does not support unit style lj. Please "
               "use other "
               "unit styles like metal or real unit instead.");
  }

  if (narg < 3) {
    error->all(FLERR, "Illegal compute deepdos command");
  }

  // parse args
  std::string model_file = std::string(arg[2]);
  std::string keyword = (narg >= 4) ? std::string(arg[3]) : "atom";

  if (keyword == "atom") {
    mode = ATOM;
  } else if (keyword == "total") {
    mode = TOTAL;
  } else {
    error->all(FLERR,
               "Illegal compute deepdos command: unknown mode. "
               "Must be 'atom' or 'total'.");
  }

  // initialize deepdos model
  int gpu_rank = dp.get_node_rank();
  try {
    dd.init(model_file, gpu_rank);
  } catch (deepmd_compat::deepmd_exception& e) {
    error->one(FLERR, e.what());
  }
  numb_dos = dd.numb_dos();

  if (mode == ATOM) {
    peratom_flag = 1;
    size_peratom_cols = numb_dos;
    pressatomflag = 0;
  } else {
    vector_flag = 1;
    size_vector = numb_dos;
    extvector = 1;
  }

  timeflag = 1;
  nmax = 0;

  dist_unit_cvt_factor = force->angstrom;
}

/* ---------------------------------------------------------------------- */

ComputeDeepdos::~ComputeDeepdos() {
  memory->destroy(dos_atom_array);
  memory->destroy(dos_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeDeepdos::init() {
  // need an occasional full neighbor list
#if LAMMPS_VERSION_NUMBER >= 20220324
  neighbor->add_request(this,
                        NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
#else
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
#endif
}

void ComputeDeepdos::init_list(int /*id*/, NeighList* ptr) {
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeDeepdos::compute_peratom() {
  invoked_peratom = update->ntimestep;

  // grow local tensor array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(dos_atom_array);
    nmax = atom->nmax;
    memory->create(dos_atom_array, nmax, size_peratom_cols,
                   "deepdos:dos_atom_array");
    array_atom = dos_atom_array;
  }

  double** x = atom->x;
  int* type = atom->type;
  int* mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  std::vector<VALUETYPE> dcoord(nall * 3, 0.);
  std::vector<VALUETYPE> dbox(9, 0);
  std::vector<int> dtype(nall);
  // get type
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type[ii] - 1;
  }
  // get box
  dbox[0] = domain->h[0] / dist_unit_cvt_factor;  // xx
  dbox[4] = domain->h[1] / dist_unit_cvt_factor;  // yy
  dbox[8] = domain->h[2] / dist_unit_cvt_factor;  // zz
  dbox[7] = domain->h[3] / dist_unit_cvt_factor;  // zy
  dbox[6] = domain->h[4] / dist_unit_cvt_factor;  // zx
  dbox[3] = domain->h[5] / dist_unit_cvt_factor;  // yx
  // get coord
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dcoord[ii * 3 + dd] =
          (x[ii][dd] - domain->boxlo[dd]) / dist_unit_cvt_factor;
    }
  }

  // invoke full neighbor list (will copy or build if necessary)
  neighbor->build_one(list);
  deepmd_compat::InputNlist lmp_list(list->inum, list->ilist, list->numneigh,
                                     list->firstneigh);
  lmp_list.set_mask(NEIGHMASK);
  lmp_list.set_mapping(list->mapping);

  // declare outputs
  std::vector<VALUETYPE> dos, atom_dos;

  // compute DOS
  try {
    dd.compute(dos, atom_dos, dcoord, dtype, dbox, nghost, lmp_list);
  } catch (deepmd_compat::deepmd_exception& e) {
    error->one(FLERR, e.what());
  }

  // store the per-atom result in dos_atom_array
  for (int ii = 0; ii < nlocal; ++ii) {
    bool ingroup = (mask[ii] & groupbit);
    if (ingroup) {
      for (int jj = 0; jj < numb_dos; ++jj) {
        dos_atom_array[ii][jj] = atom_dos[ii * numb_dos + jj];
      }
    } else {
      for (int jj = 0; jj < numb_dos; ++jj) {
        dos_atom_array[ii][jj] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeDeepdos::compute_vector() {
  invoked_vector = update->ntimestep;

  double** x = atom->x;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  std::vector<VALUETYPE> dcoord(nall * 3, 0.);
  std::vector<VALUETYPE> dbox(9, 0);
  std::vector<int> dtype(nall);
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type[ii] - 1;
  }
  dbox[0] = domain->h[0] / dist_unit_cvt_factor;
  dbox[4] = domain->h[1] / dist_unit_cvt_factor;
  dbox[8] = domain->h[2] / dist_unit_cvt_factor;
  dbox[7] = domain->h[3] / dist_unit_cvt_factor;
  dbox[6] = domain->h[4] / dist_unit_cvt_factor;
  dbox[3] = domain->h[5] / dist_unit_cvt_factor;
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dcoord[ii * 3 + dd] =
          (x[ii][dd] - domain->boxlo[dd]) / dist_unit_cvt_factor;
    }
  }

  neighbor->build_one(list);
  deepmd_compat::InputNlist lmp_list(list->inum, list->ilist, list->numneigh,
                                     list->firstneigh);
  lmp_list.set_mask(NEIGHMASK);
  lmp_list.set_mapping(list->mapping);

  std::vector<VALUETYPE> dos, atom_dos;
  try {
    dd.compute(dos, atom_dos, dcoord, dtype, dbox, nghost, lmp_list);
  } catch (deepmd_compat::deepmd_exception& e) {
    error->one(FLERR, e.what());
  }

  // store the total dos result
  if (dos.empty()) {
    // model doesn't provide global dos; compute by summing atom_dos
    memory->destroy(dos_vector);
    memory->create(dos_vector, size_vector, "deepdos:dos_vector");
    for (int ii = 0; ii < numb_dos; ++ii) {
      double sum = 0.0;
      for (int jj = 0; jj < nlocal; ++jj) {
        sum += atom_dos[jj * numb_dos + ii];
      }
      MPI_Allreduce(&sum, &dos_vector[ii], 1, MPI_DOUBLE, MPI_SUM, world);
    }
  } else {
    memory->destroy(dos_vector);
    memory->create(dos_vector, size_vector, "deepdos:dos_vector");
    for (int ii = 0; ii < numb_dos; ++ii) {
      dos_vector[ii] = dos[ii];
    }
  }
  vector = dos_vector;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeDeepdos::memory_usage() {
  double bytes = 0;
  if (mode == ATOM) {
    bytes += static_cast<size_t>(nmax) * size_peratom_cols * sizeof(double);
  } else {
    bytes += static_cast<size_t>(size_vector) * sizeof(double);
  }
  return bytes;
}

#else

using namespace LAMMPS_NS;

ComputeDeepdos::ComputeDeepdos(LAMMPS* lmp, int narg, char** arg)
    : Compute(lmp, narg, arg),
      dp(lmp),
      dos_atom_array(nullptr),
      dos_vector(nullptr),
      list(nullptr) {
  error->all(FLERR,
             "Compute deepdos requires the C++ API (DP_USE_CXX_API).");
}

ComputeDeepdos::~ComputeDeepdos() {
  memory->destroy(dos_atom_array);
  memory->destroy(dos_vector);
}

void ComputeDeepdos::init() {}
void ComputeDeepdos::compute_peratom() {}
void ComputeDeepdos::compute_vector() {}
double ComputeDeepdos::memory_usage() { return 0; }
void ComputeDeepdos::init_list(int, class NeighList*) {}

#endif
