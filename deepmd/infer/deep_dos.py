import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from deepmd.common import make_default_mesh
from deepmd.env import default_tf_session_config, tf
from deepmd.infer.data_modifier import DipoleChargeModifier
from deepmd.infer.deep_eval import DeepEval

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class DeepDOS(DeepEval):
    """Constructor.
    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    Warnings
    --------
    For developers: `DeepTensor` initializer must be called at the end after
    `self.tensors` are modified because it uses the data in `self.tensors` dict.
    Do not chanage the order!
    """

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False
    ) -> None:

        # add these tensors on top of what is defined by DeepTensor Class
        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # descrpt attrs
                "t_ntypes": "descrpt_attr/ntypes:0",
                "t_rcut": "descrpt_attr/rcut:0",
                # fitting attrs
                "t_dfparam": "fitting_attr/dfparam:0",
                "t_daparam": "fitting_attr/daparam:0",
                "t_numb_dos":"fitting_attr/numb_dos:0",
                # model attrs
                "t_tmap": "model_attr/tmap:0",
                # inputs
                "t_coord": "t_coord:0",
                "t_type": "t_type:0",
                "t_natoms": "t_natoms:0",
                "t_box": "t_box:0",
                "t_mesh": "t_mesh:0",
                # add output tensors
                "t_dos": "o_dos:0",
                "t_ados": "o_atom_dos:0"
            },
        )
        DeepEval.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph
        )

        # load optional tensors
        operations = [op.name for op in self.graph.get_operations()]
        # check if the graph has these operations:
        # if yes add them

        if 'load/t_fparam' in operations:
            self.tensors.update({"t_fparam": "t_fparam:0"})
            self.has_fparam = True
        else:
            log.debug(f"Could not get tensor 't_fparam:0'")
            self.t_fparam = None
            self.has_fparam = False

        if 'load/t_aparam' in operations:
            self.tensors.update({"t_aparam": "t_aparam:0"})
            self.has_aparam = True
        else:
            log.debug(f"Could not get tensor 't_aparam:0'")
            self.t_aparam = None
            self.has_aparam = False

        # now load tensors to object attributes
        for attr_name, tensor_name in self.tensors.items():
            self._get_tensor(tensor_name, attr_name)

        # start a tf session associated to the graph
        self.sess = tf.Session(graph=self.graph, config=default_tf_session_config)
        self._run_default_sess()
        self.tmap = self.tmap.decode('UTF-8').split()        

        # setup modifier
        try:
            t_modifier_type = self._get_tensor("modifier_attr/type:0")
            self.modifier_type = self.sess.run(t_modifier_type).decode("UTF-8")
        except (ValueError, KeyError):
            self.modifier_type = None

    def _run_default_sess(self):
        [self.ntypes, self.rcut, self.numb_dos, self.dfparam, self.daparam, self.tmap] = self.sess.run(
            [self.t_ntypes, self.t_rcut, self.t_numb_dos, self.t_dfparam, self.t_daparam, self.t_tmap]
        )

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return self.ntypes

    def get_rcut(self) -> float:
        """Get the cut-off radius of this model."""
        return self.rcut
    
    def get_numb_dos(self) -> int:
        """
        Get the number of density of state of this DP
        """
        return self.numb_dos
    
    def get_type_map(self) -> List[int]:
        """Get the type map (element name of the atom types) of this model."""
        return self.tmap

    def get_sel_type(self) -> List[int]:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dfparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.daparam

    def eval(
        self,
        coords: np.array,
        cells: np.array,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None
    ) -> Tuple[np.ndarray, ...]:
        """Evaluate the dos, ados by using this DP.
        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Calculate the atomic energy and virial
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.
        Returns
        -------
        dos
            The electron density of state.
        ados
            The atom-sited density of state. Only returned when atomic == True
        """
        if atomic:
            if self.modifier_type is not None:
                raise RuntimeError('modifier does not support atomic modification')
            return self._eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic)
        else :
            dos = self._eval_inner(coords, cells, atom_types, fparam = fparam, aparam = aparam, atomic = atomic)
            return dos

    def _eval_inner(
        self,
        coords,
        cells,
        atom_types,
        fparam=None,
        aparam=None,
        atomic=False
    ):
        # standarize the shape of inputs
        atom_types = np.array(atom_types, dtype = int).reshape([-1])
        natoms = atom_types.size
        coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        if cells is None:
            pbc = False
            # make cells to work around the requirement of pbc
            cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
        else:
            pbc = True
            cells = np.array(cells).reshape([nframes, 9])
        
        if self.has_fparam :
            assert(fparam is not None)
            fparam = np.array(fparam)
        if self.has_aparam :
            assert(aparam is not None)
            aparam = np.array(aparam)

        # reshape the inputs 
        if self.has_fparam :
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim :
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim :
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d or %d' % (nframes, fdim, fdim))
        if self.has_aparam :
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d' % (nframes, natoms, fdim, natoms, fdim, fdim))

        # sort inputs
        coords, atom_types, imap = self.sort_input(coords, atom_types)         

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)

        # evaluate
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = np.tile(atom_types, [nframes, 1]).reshape([-1])
        t_out = [self.t_dos]
        if atomic :
            t_out += [self.t_ados]

        feed_dict_test[self.t_coord] = np.reshape(coords, [-1])
        feed_dict_test[self.t_box  ] = np.reshape(cells , [-1])
        if pbc:
            feed_dict_test[self.t_mesh ] = make_default_mesh(cells)
        else:
            feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)
        if self.has_fparam:
            feed_dict_test[self.t_fparam] = np.reshape(fparam, [-1])
        if self.has_aparam:
            feed_dict_test[self.t_aparam] = np.reshape(aparam, [-1])
        v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
        dos = v_out[0]
        if atomic:
            ados = v_out[1]

        # reverse map of the outputs
        if atomic :
            ados  = self.reverse_map(np.reshape(ados, [nframes,-1,self.numb_dos]), imap)

        dos = np.reshape(dos, [nframes, self.numb_dos])
        if atomic:
            ados = np.reshape(ados, [nframes, natoms, self.numb_dos])
            return dos, ados
        else :
            return dos
