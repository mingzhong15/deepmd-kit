// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <memory>

#include "common.h"
#include "neighbor_list.h"

namespace deepmd {

class DeepDOSBase {
 public:
  DeepDOSBase(){};
  virtual ~DeepDOSBase(){};

  virtual void init(const std::string& model,
                    const int& gpu_rank = 0,
                    const std::string& file_content = "") = 0;

  virtual double cutoff() const = 0;
  virtual int numb_types() const = 0;
  virtual int numb_dos() const = 0;
  virtual void get_type_map(std::string& type_map) = 0;

  virtual void computew(std::vector<double>& dos,
                        std::vector<double>& atom_dos,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const int nghost,
                        const InputNlist& inlist) = 0;
  virtual void computew(std::vector<float>& dos,
                        std::vector<float>& atom_dos,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const int nghost,
                        const InputNlist& inlist) = 0;
};

class DeepDOS {
 public:
  DeepDOS();
  virtual ~DeepDOS();
  DeepDOS(const std::string& model,
          const int& gpu_rank = 0,
          const std::string& file_content = "");

  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

  void print_summary(const std::string& pre) const;

  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& dos,
               std::vector<VALUETYPE>& atom_dos,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist);

  double cutoff() const;
  int numb_types() const;
  int numb_dos() const;
  void get_type_map(std::string& type_map);

 protected:
  bool inited;
  std::shared_ptr<deepmd::DeepDOSBase> dos;
};

}  // namespace deepmd
