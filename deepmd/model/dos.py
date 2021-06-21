import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.utils.pair_tab import PairTab
from deepmd.common import ClassArg
from deepmd.env import global_cvt_2_ener_float, MODEL_VERSION
from deepmd.env import op_module
from .model_stat import make_stat_input, merge_sys_stat

class DOSModel() :
    model_type = 'dos'

    def __init__ (
            self, 
            descrpt, 
            fitting, 
            type_map : List[str] = None,
            data_stat_nbatch : int = 10,
            data_stat_protect : float = 1e-2
    ) -> None:
        """
        Constructor
        Parameters
        ----------
        descrpt
                Descriptor
        fitting
                Fitting net
        type_map
                Mapping atom type to the name (str) of the type.
                For example `type_map[1]` gives the name of the type 1.
        data_stat_nbatch
                Number of frames used for data statistic
        data_stat_protect
                Protect parameter for atomic energy regression
        """
        # descriptor
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # fitting
        self.fitting = fitting
        self.numb_dos = self.fitting.get_numb_dos()
        self.numb_fparam = self.fitting.get_numb_fparam()
        # other inputs
        if type_map is None:
            self.type_map = []
        else:
            self.type_map = type_map
        self.data_stat_nbatch = data_stat_nbatch
        self.data_stat_protect = data_stat_protect
        
    def get_numb_dos(self):
        return self.numb_dos
    
    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_type_map (self) :
        return self.type_map

    def data_stat(self, data):
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys = False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(m_all_stat, protection = self.data_stat_protect)
        # self._compute_output_stat(all_stat)
        # self.bias_atom_e = data.compute_energy_shift(self.rcond)

    def _compute_input_stat (self, all_stat, protection = 1e-2) :
        self.descrpt.compute_input_stats(all_stat['coord'],
                                         all_stat['box'],
                                         all_stat['type'],
                                         all_stat['natoms_vec'],
                                         all_stat['default_mesh'], 
                                         all_stat)
        self.fitting.compute_input_stats(all_stat, protection = protection)

    def _compute_output_stat (self, all_stat) :
        self.fitting.compute_output_stats(all_stat)

    
    def build (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):

        with tf.variable_scope('model_attr' + suffix, reuse = reuse) :
            t_tmap = tf.constant(' '.join(self.type_map), 
                                 name = 'tmap', 
                                 dtype = tf.string)
            t_mt = tf.constant(self.model_type, 
                               name = 'model_type', 
                               dtype = tf.string)
            t_ver = tf.constant(MODEL_VERSION,
                                name = 'model_version',
                                dtype = tf.string)
            t_od =  tf.constant(self.get_numb_dos(), 
                               name = 'output_dim', 
                               dtype = tf.int32)

        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        dout \
            = self.descrpt.build(coord_,
                                 atype_,
                                 natoms,
                                 box,
                                 mesh,
                                 input_dict,
                                 suffix = suffix,
                                 reuse = reuse)
        dout = tf.identity(dout, name='o_descriptor')

        atom_dos = self.fitting.build (dout, 
                                        natoms, 
                                        input_dict, 
                                        reuse = reuse, 
                                        suffix = suffix)


        dos_raw = atom_dos

        dos_raw = tf.reshape(dos_raw, [natoms[0], -1], name = 'o_atom_dos'+suffix)
        dos = tf.reduce_sum(global_cvt_2_ener_float(dos_raw), axis=0, name='o_dos'+suffix)

        model_dict = {}
        model_dict['dos'] = dos
        model_dict['atom_dos'] = dos_raw
        model_dict['coord'] = coord
        model_dict['atype'] = atype
        
        return model_dict
