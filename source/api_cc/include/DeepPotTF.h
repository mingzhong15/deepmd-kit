// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <type_traits>
#include <vector>

#include "DeepPot.h"
#include "common.h"
#include "commonTF.h"
#include "neighbor_list.h"

namespace deepmd {
/**
 * @brief TensorFlow implementation for Deep Potential.
 **/
class DeepPotTF : public DeepPotBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPotTF();
  virtual ~DeepPotTF();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepPotTF(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

 private:
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);
  /**
   * @brief Evaluate the energy, force, and virial with the mixed type
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The array should be of size nframes x
   *natoms.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(ENERGYVTYPE& ener,
                          std::vector<VALUETYPE>& force,
                          std::vector<VALUETYPE>& virial,
                          std::vector<VALUETYPE>& atom_energy,
                          std::vector<VALUETYPE>& atom_virial,
                          const int& nframes,
                          const std::vector<VALUETYPE>& coord,
                          const std::vector<int>& atype,
                          const std::vector<VALUETYPE>& box,
                          const std::vector<VALUETYPE>& fparam,
                          const std::vector<VALUETYPE>& aparam,
                          const bool atomic);

 public:
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  /**
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const {
    assert(inited);
    return ntypes_spin;
  };
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const {
    assert(inited);
    return dfparam;
  };
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const {
    assert(inited);
    return daparam;
  };
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string& type_map);

  /**
   * @brief Get whether the atom dimension of aparam is nall instead of fparam.
   * @param[out] aparam_nall whether the atom dimension of aparam is nall
   *instead of fparam.
   **/
  bool is_aparam_nall() const {
    assert(inited);
    return aparam_nall;
  };
  /**
   * @brief Check if the model has default frame parameters.
   * @return Always false for TF backend.
   **/
  bool has_default_fparam() const { return false; };

  /**
   * @brief Check if the model produced electronic entropy in the last call.
   * @return true if ele_entropy is available (model uses frame parameters).
   **/
  bool has_ele_entropy() const override { return !ele_entropy_.empty(); }
  /**
   * @brief Get the electronic entropy computed in the last call.
   * @return The electronic entropy (empty if not computed).
   **/
  const std::vector<double>& get_ele_entropy() const override {
    return ele_entropy_;
  }
  /**
   * @brief Get the free energy (Helmholtz) computed in the last call.
   * @return The free energy, equal to the model energy.
   **/
  const std::vector<double>& get_free_energy() const override {
    return free_energy_;
  }
  /**
   * @brief Get the internal energy computed in the last call.
   * @return The internal energy = energy + fparam * ele_entropy.
   **/
  const std::vector<double>& get_internal_energy() const override {
    return internal_energy_;
  }

  // forward to template class
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<double>& force,
                           std::vector<double>& virial,
                           std::vector<double>& atom_energy,
                           std::vector<double>& atom_virial,
                           const int& nframes,
                           const std::vector<double>& coord,
                           const std::vector<int>& atype,
                           const std::vector<double>& box,
                           const std::vector<double>& fparam,
                           const std::vector<double>& aparam,
                           const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<float>& force,
                           std::vector<float>& virial,
                           std::vector<float>& atom_energy,
                           std::vector<float>& atom_virial,
                           const int& nframes,
                           const std::vector<float>& coord,
                           const std::vector<int>& atype,
                           const std::vector<float>& box,
                           const std::vector<float>& fparam,
                           const std::vector<float>& aparam,
                           const bool atomic);

 private:
  tensorflow::Session* session;
  int num_intra_nthreads, num_inter_nthreads;
  tensorflow::GraphDef* graph_def;
  bool inited;
  template <class VT>
  VT get_scalar(const std::string& name) const;
  double rcut;
  int dtype;
  double cell_size;
  std::string model_type;
  std::string model_version;
  int ntypes;
  int ntypes_spin;
  int dfparam;
  int daparam;
  bool aparam_nall;
  // electronic entropy and derived thermodynamic quantities
  std::vector<double> ele_entropy_;
  std::vector<double> free_energy_;
  std::vector<double> internal_energy_;
  /**
   * @brief Validate the size of frame and atomic parameters.
   * @param[in] nframes The number of frames.
   * @param[in] nloc The number of local atoms.
   * @param[in] fparam The frame parameter.
   * @param[in] aparam The atomic parameter.
   * @tparam VALUETYPE The type of the parameters, double or float.
   */
  template <typename VALUETYPE>
  void validate_fparam_aparam(const int& nframes,
                              const int& nloc,
                              const std::vector<VALUETYPE>& fparam,
                              const std::vector<VALUETYPE>& aparam) const;
  /**
   * @brief Tile the frame or atomic parameters if there is only
   * a single frame of frame or atomic parameters.
   * @param[out] out_param The tiled frame or atomic parameters.
   * @param[in] nframes The number of frames.
   * @param[in] dparam The dimension of the frame or atomic parameters in a
   * frame.
   * @param[in] param The frame or atomic parameters.
   * @tparam VALUETYPE The type of the parameters, double or float.
   */
  template <typename VALUETYPE>
  void tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
                          const int& nframes,
                          const int& dparam,
                          const std::vector<VALUETYPE>& param) const;
  // copy neighbor list info from host
  bool init_nbor;
  std::vector<int> sec_a;
  NeighborListData nlist_data;
  InputNlist nlist;
  AtomMap atommap;
  /**
   * @brief Post-process electronic entropy into member variables.
   * Stores ele_entropy_, free_energy_ (= energy), and internal_energy_
   * (= energy + sum_j fparam[j] * ele_entropy[j]) per frame. This is
   * called after run_model with the returned dele_entropy vector and
   * the per-frame fparam; both may be empty, in which case only
   * free_energy_ is set.
   * @param[in] ener Per-frame model energy.
   * @param[in] dele_entropy Per-frame electronic entropy (flattened
   *                        to nf*nfp), or empty if not available.
   * @param[in] fparam Per-frame frame parameters (nf*nfp), or empty.
   * @param[in] nframes Number of frames.
   **/
  void _post_ele_entropy(const std::vector<double>& ener,
                         const std::vector<ENERGYTYPE>& dele_entropy,
                         const std::vector<double>& fparam,
                         const int nframes);
  /**
   * @brief Post-process electronic entropy into member variables.
   * Stores ele_entropy_, free_energy_ (= energy), and internal_energy_
   * (= energy + sum_j fparam[j] * ele_entropy[j]) per frame. Called
   * after run_model with the returned dele_entropy and per-frame fparam;
   * both may be empty, in which case only free_energy_ is set.
   * @tparam FPARAMVT The frame parameter value type (double or float).
   **/
  template <typename FPARAMVT>
  void _post_ele_entropy_tmpl(
      const std::vector<ENERGYTYPE>& dener_vec,
      const std::vector<ENERGYTYPE>& dele_entropy,
      const std::vector<FPARAMVT>& fparam,
      const int nframes) {
    std::vector<double> ener_vec(dener_vec.begin(), dener_vec.end());
    std::vector<double> fparam_d(fparam.begin(), fparam.end());
    _post_ele_entropy(ener_vec, dele_entropy, fparam_d, nframes);
  }
  /**
   * @brief Convert ENERGYVTYPE (scalar or vector) to a per-frame
   * std::vector<ENERGYTYPE> for use with _post_ele_entropy_tmpl.
   * The scalar overload wraps into a 1-element vector; the vector
   * overload is a no-op copy. Dispatched via std::is_same SFINAE.
   **/
  template <typename ENERGYVTYPE>
  static typename std::enable_if<
      std::is_same<ENERGYVTYPE, ENERGYTYPE>::value,
      std::vector<ENERGYTYPE>>::type
  _ener_to_vec(const ENERGYVTYPE& dener, int /*nframes*/) {
    return std::vector<ENERGYTYPE>(1, dener);
  }
  template <typename ENERGYVTYPE>
  static typename std::enable_if<
      std::is_same<ENERGYVTYPE, std::vector<ENERGYTYPE>>::value,
      std::vector<ENERGYTYPE>>::type
  _ener_to_vec(const ENERGYVTYPE& dener, int /*nframes*/) {
    return dener;
  }
};

}  // namespace deepmd
