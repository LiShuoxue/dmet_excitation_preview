import sys, os
from pyscf import fci, lib
import numpy as np
from numpy import einsum
from scipy import linalg as la
from collections.abc import Iterable
from pyscf.fci.addons import _unpack_nelec
from dmet_excitation.chain_lattice import ChainLattice


# Constants

DEFAULT_OP_SCHEMES = {
    '1-ee-P': ['P'],
    '2-ee-P': ['PP'],
    '3-ee-P': ['PPP'],
    '2-ee-PG': ['PP', '+-', '-+'],
    '3-ee-PG': ['PPP', 'P+-', 'P-+', '+-P', '-+P', '+P-', '-P+'],
    '2-ee-full': ['PP', '+-', '-+', 'cd', 'dc', 'CD', 'DC', 'EF', 'FE'],
    '2-ip-full': ['Pd', 'dP', '-D', 'D-'],
    '3-ip-P': ['PPd', 'PdP', 'dPP'],
    '3-ip-PG': ['PPd', 'PdP', 'dPP',
                 '+-d', '+d-', 'd+-', '-+d', '-d+', '-+d',
                 'P-D', 'D-P', '-PD', '-DP', 'PD-', 'DP-'],
}

# name, N, 2S, numbers, operators
SITE_OPS_SCHEME_INFO = {
        "P": ( 0,  0, 4, ('cdCD', 'cdDC', 'dcCD', 'dcDC')), # nn, nm, mn, mm
        "c": ( 1,  1, 2, ('cCD', 'cDC')),
        "C": ( 1, -1, 2, ('cdC', 'dcC')),
        "d": (-1, -1, 2, ('dCD', 'dDC')),
        "D": (-1,  1, 2, ('cdD', 'dcD')),
        "E": ( 2,  0, 1, ('cC', )),
        "F": (-2,  0, 1, ('dD', )),
        "+": ( 0,  2, 1, ('cD', )),
        "-": ( 0, -2, 1, ('dC', ))
        }


# DMET Functions

def get_bath(rdm1, imp_idx):
    imp_idx = np.asarray(imp_idx, dtype=int)
    nimp = len(imp_idx)
    nlo = rdm1.shape[1]
    env_idx = np.asarray([idx for idx in range(nlo) if idx not in imp_idx], dtype=int)
    rdm1_imp_imp = rdm1[imp_idx][:, imp_idx]
    rdm1_env_imp = rdm1[env_idx][:, imp_idx]
    dw, dv = la.eigh(rdm1_imp_imp / 2.)
    C_bath = einsum("Ri,ie,e->Re", rdm1_env_imp / 2., dv, 1. / np.sqrt(dw * (1. - dw)))
    C_lo_eo = np.zeros((rdm1.shape[1], nimp * 2))
    C_lo_eo[imp_idx, :nimp] += np.eye(nimp)
    C_lo_eo[env_idx, nimp:] += C_bath
    return C_lo_eo


def get_h2_emb(eri, C_lo_eo):
    operands = (eri, ) + (C_lo_eo, ) * 4
    return np.einsum("ijkl,iI,jJ,kK,lL->IJKL", *operands)


def get_h1_emb(fock_lo, rdm1_lo, C_lo_eo, h2_emb):
    fock_emb = C_lo_eo.T @ fock_lo @ C_lo_eo
    rdm1_emb_mf = C_lo_eo.T @ rdm1_lo @ C_lo_eo
    _eri_asymm = h2_emb - .5 * h2_emb.transpose((0, 3, 2, 1))
    vhf_dc = np.einsum("ijkl,kl->ij", _eri_asymm, rdm1_emb_mf)
    return fock_emb - vhf_dc

    
def get_weight_matrix(imp_idx, env_idx, neo, ndim=4):
    idxlst = (env_idx, imp_idx)
    w = np.zeros((neo, ) * ndim)
    for bools in np.ndindex((2, ) * ndim):
        weight = sum(bools) / float(ndim)
        w[np.ix_(*map(idxlst.__getitem__, bools))] += weight
    return w


def get_h1_scaled(h1, imp_idx, env_idx):
    return h1 * get_weight_matrix(imp_idx, env_idx, neo=h1.shape[-1], ndim=2)


def get_h2_scaled(h2, imp_idx, env_idx):
    return h2 * get_weight_matrix(imp_idx, env_idx, neo=h2.shape[-1], ndim=4)


def get_ham_dmet(hcore_lo, veff_lo, h2_emb, rdm1_emb, imp_idx, C_lo_eo):
    _eri_asymm = h2_emb - .5 * h2_emb.transpose((0, 3, 2, 1))
    neo = rdm1_emb.shape[-1]
    env_idx = np.asarray([idx for idx in range(neo) if idx not in imp_idx], dtype=int)
    h1_scaled = C_lo_eo.T @ hcore_lo @ C_lo_eo
    veff_emb = C_lo_eo.T @ veff_lo @ C_lo_eo
    veff_dc = np.einsum("ijkl,kl->ij", _eri_asymm, rdm1_emb)
    veff_emb  -= veff_dc
    h1_scaled += veff_emb * 0.5
    h1_scaled = get_h1_scaled(h1_scaled, imp_idx, env_idx)
    h2_scaled = get_h2_scaled(h2_emb, imp_idx, env_idx)
    return (h1_scaled, h2_scaled)


class Operator_tuple(tuple):
    """Save the information of site operators"""

    def operator_mul(self, op_list2):
        results = list()
        for op1, val1 in self:
            for op2, val2 in op_list2:
                op_new = op1 + "|" + op2
                val = val1 * val2
                results.append((op_new, val))
        return Operator_tuple(results)
    
    def operator_scaling(self, scale):
        results = list()
        for op, val in self:
            results.append((op, val*scale))
        return Operator_tuple(results)

    def conj(self):
        raise NotImplementedError

    @property
    def nsite(self):
        return len(self[0][0].split('|'))

    @property
    def dnelec(self):   # (up, down)
        dneleca, dnelecb = 0, 0
        op_name = self[0][0]
        for astr in op_name:
            if astr == 'c': dneleca += 1
            if astr == 'C': dnelecb += 1
            if astr == 'd': dneleca -= 1
            if astr == 'D': dnelecb -= 1
        return (dneleca, dnelecb)

    def __mul__(self, other):
        if isinstance(other, Operator_tuple):
            return self.operator_mul(other)
        elif not isinstance(other, Iterable):
            return self.operator_scaling(other)
        else:
            raise ValueError("Type of the multiplied object other than Operator_tuple or scalar not identified!")

    __rmul__ = __mul__

    def __add__(self, other):
        assert isinstance(other, Operator_tuple)
        results = list(self)
        for op, val in other:
            if np.abs(val)>1e-10:
                added = False
                for idx, (op0, val0) in enumerate(results):
                    if op == op0:
                        added = True
                        results[idx] = (op, val+val0)
                if not added:
                    results.append((op, val))
        return Operator_tuple(results)
    
    def __str__(self):
        astr = str()
        for i, (op, val) in enumerate(self):
            astr += "%8.5f * %-13s" % (val, op)
            if i != len(self)-1:
                astr += " + "
        return astr


def op_sz_to_fci(op_tuple:Operator_tuple, site_idxs:Iterable, 
                 civec:np.ndarray, norb:int, nelec:int) -> np.ndarray:
    """
    for a FCI vector, apply the excitation operator and get the new FCI vector
    """
    op_dict = {
        "c": fci.addons.cre_a,
        "d": fci.addons.des_a,
        "C": fci.addons.cre_b,
        "D": fci.addons.des_b
    }

    delec = {
        "c": (1, 0), "d": (-1, 0),
        "C": (0, 1), "D": (0, -1),
    }

    res_tot = 0.
    for op, val in op_tuple:
        op_split = op.split("|")
        op_str_to_mpo = str()
        site_idx_to_mpo = list()
        for idx, op_str_site in enumerate(op_split):
            for astr in op_str_site:
                if astr in op_dict.keys():  # cdCD
                    site_idx = site_idxs[idx]
                    op_str_to_mpo += astr
                    site_idx_to_mpo.append(site_idx)
        nelec_a, nelec_b = fci.addons._unpack_nelec(nelec)
        res_ci = civec.copy()
        for idx in range(len(op_str_to_mpo)-1, -1, -1):
            op_str = op_str_to_mpo[idx]
            site   = site_idx_to_mpo[idx]
            res_ci = op_dict[op_str](res_ci, norb, (nelec_a,nelec_b), ap_id=site)
            nelec_a += delec[op_str][0]
            nelec_b += delec[op_str][1]
        res_tot += res_ci * val
    return res_tot


def _Hc(op:tuple, ket:np.ndarray, norb:int, nelec:int):
    """
    Contract the 

    op: h1e and g2e operator
    ket: fci vector
    """
    h1e, eri = op

    if h1e is not None:
        if len(np.array(h1e).shape) == 2:
            fci_cls = fci.direct_spin1
        elif len(np.array(h1e).shape) == 3:
            fci_cls = fci.direct_uhf
        else:
            raise ValueError('in excited.tangent_space.fci._Hc: h1e should have dimension 2 or 3.')
    
    if eri is not None:
        if len(np.array(eri).shape) in (2, 4):
            fci_cls = fci.direct_spin1
        elif len(np.array(eri).shape) in (3, 5):
            fci_cls = fci.direct_uhf
        else:
            raise ValueError('in excited.tangent_space.fci._Hc: eri should have dimension in (2,3,4,5).')
    
    if h1e is not None:
        if eri is not None:
            eri = fci_cls.absorb_h1e(h1e, eri, norb, nelec, fac=.5)
    op_ket = 0
    if eri is None:
        if h1e is not None:
            op_ket = fci_cls.contract_1e(h1e, ket, norb, nelec)
    else:
        op_ket = fci_cls.contract_2e(eri, ket, norb, nelec)
    return op_ket


def _expectation(bra, ket, op, norb, nelec):
    # <bra|op|ket>
    # nelec: same as the ket
    if op is None:
        return np.einsum("ij,ij->", bra.conj(), ket)
    else:
        op_ket = _Hc(op, ket, norb, nelec)
        if np.allclose(op_ket, 0.):
            return 0
        else:
            return np.einsum("ij,ij->", bra.conj(), op_ket)


def build_patch_vectors_from_ops(
    civec,
    sites:Iterable,
    ops:Iterable,
    norb:int,
    nelec:int,
    scratch="./tempdir",
    tag="patch",
    log:lib.logger.Logger=None
):  
    """nelec: the initial number of electron before operator"""
    
    if scratch[-1] == '/':
        scratch = scratch[:-1]

    for iop, op in enumerate(ops):
        dket = op_sz_to_fci(op, sites, civec, norb, nelec)
        log.debug("\n excitation operator %s saved in %s-%-4d" % (op, tag, iop))
        np.save("%s/%s-%d.npy"%(scratch, tag, iop), dket)


def _contract(scratch, tag_bra, tag_ket, dim_bra, dim_ket,
              op, norb, nelec, log:lib.logger.Logger):
    # Shuoxue NOTE: nelec same as the ket
    if scratch[-1] == '/':
        scratch = scratch[:-1]
    op_mat = np.zeros((dim_bra, dim_ket))
    log.debug("\t Solving the expectation for <%s|%s>", tag_bra, tag_ket)
    for id_bra in range(dim_bra):
        bra = np.load("%s/%s-%d.npy"%(scratch, tag_bra, id_bra))
        for id_ket in range(dim_ket):
            ket = np.load("%s/%s-%d.npy"%(scratch, tag_ket, id_ket))            
            op_mat[id_bra, id_ket] = _expectation(bra, ket, op, norb, nelec)
    return op_mat


def _contract_with_gs(scratch:str, tag_ket:str, tag_gs:str, dim_ket:int, op, norb, nelec, log):
    if scratch[-1] == '/':
        scratch = scratch[:-1]
    op_mat = np.zeros((dim_ket, ))
    log.debug("\t Solving the expectation for <GS %s|KET %s>", tag_gs, tag_ket)
    gs = np.load('%s/%s.npy' % (scratch, tag_gs))
    for id_ket in range(dim_ket):
        ket = np.load("%s/%s-%d.npy"%(scratch, tag_ket, id_ket))
        op_mat[id_ket] = _expectation(gs, ket, op, norb, nelec)
    return op_mat


def calc_norm(norb, nelec, scratch, tag, dim, log=None):
    if scratch[-1] == '/':
        scratch = scratch[:-1]
    norm_arr = np.zeros(dim)
    for id_ket in range(dim):
        ket = np.load("%s/%s-%d.npy"%(scratch, tag, id_ket))
        norm = np.sqrt(_expectation(ket, ket, None, norb, nelec))
        if np.allclose(norm, 0.):
            norm = 1.
        norm_arr[id_ket] += norm
    return norm_arr


def get_operators_from_scheme(schemes:str):
    """
    Get the operators from the operator schemes.
    """
    PTR_NPATT = 2
    PTR_PATT = 3
    ops = list()
    for scheme in schemes:
        npatts = np.ndindex(*[SITE_OPS_SCHEME_INFO[s][PTR_NPATT] for s in scheme])
        for inpatt, npatt in enumerate(npatts):
            patts = [SITE_OPS_SCHEME_INFO[s][PTR_PATT][npatt[i]]
                    for i, s in enumerate(scheme)]
            ops.append(Operator_tuple((("|".join(patts), 1.), )))
    return ops


class LocalExcitationBases:
    """
    Object to manage the local excitation bases.

        tag_gs : the tag of ground state
        tag_ts : the tag of excitation space
        op_scheme : the operator scheme for generalizing local excitations.

        operators : list of the local excitation operators.
    """
    def __init__(self, verbose=4, stdout=sys.stdout, **kwargs):
        # information
        self.args_info = ('tag_gs', 'tag_ts', 'op_scheme', 
                     'impidxs',
                     'norb', 'nelec', 'n_target', 'solver_kwargs',
                     'restart', 'scratch_ts', 'scratch_cd',
                     'reorder_idx', 'operators',
                     )

        self.tag_gs = str()
        self.tag_ts = str()
        self.op_scheme = None
        self.impidxs = []
        self.norb = 0

        # Shuoxue NOTE nelec is the initial electron number of ground state
        self.nelec = (0, 0)
        # Shuoxue NOTE: nelec_final: the electron number of O|gs>
        self.nelec_final = (0, 0)

        self.n_target = 0
        self.restart = True
        self.scratch_ts = ""
        self.scratch_cd = ""
        self.verbose = verbose
        self.stdout  = stdout
        self.log = lib.logger.new_logger(self, verbose=verbose)

        # properties
        self.operators: list[Operator_tuple] = list()

        input_kwargs = {key : val for key, val in kwargs.items()
                        if key in self.args_info}
        self.__dict__.update(input_kwargs)

        if len(self.operators) == 0:
            self.operators = get_operators_from_scheme(schemes=self.op_scheme)

        # nelec after excitation operators.
        self.nelec_final = (self.nelec[0]+self.operators[0].dnelec[0],
                            self.nelec[1]+self.operators[0].dnelec[1])

    @property
    def tag_ts_eom(self):
        return 'Hc-%s' % self.tag_ts

    def run(self, restart=True, calc_eom=False, ham=None, operators=None):

        if operators is None:
            operators = self.operators

        kernel_args = ('tag_gs', 'tag_ts', 'scratch_ts', 'scratch_cd',
                       'norb', 'nelec', 'verbose', 'impidxs')
        kernel_kwargs = {key: val for key, val in self.__dict__.items()
                         if key in kernel_args}
        if calc_eom:
            kernel_kwargs['tag_ts'] = self.tag_ts_eom

        return build_excitation_space(restart=restart,
                                   calc_eom=calc_eom,
                                   ham=ham,
                                   operators=operators,
                                   **kernel_kwargs)

    kernel = run

    def contract_with_bra(self, op, tag_bra, tag_ket, dim_bra, dim_ket):
        return _contract(self.scratch_ts, tag_bra, tag_ket, dim_bra, dim_ket,
                    op=op, norb=self.norb, nelec=self.nelec_final, log=self.log)

    def contract_with_gs(self, op_name, op, tag_ket, tag_gs, dim_ket, restart=True):
        gs_ts_path = '%s/%s-with_gs.npy' % (self.scratch_cd, tag_ket)

        ops = {}
        if os.path.exists(gs_ts_path):
            ops = np.load(gs_ts_path, allow_pickle=True).item()
        
        if restart:
            if op_name not in ops.keys():
                restart = False
        
        if not restart:
            res = _contract_with_gs(self.scratch_ts, tag_ket, tag_gs, dim_ket,
                    op=op, norb=self.norb, nelec=self.nelec_final, log=self.log)
            ops[op_name] = res
        else:
            res = ops[op_name]

        np.save(gs_ts_path, ops, allow_pickle=True)
        return res


class ContractData:
    def __init__(self, norm_b, norm_k, verbose, stdout):
        """
        managing the contracted block from tangent-space contraction.

        norm_b: the normalization factor for bra
        norm_k: the normalization factor for ket
        """
        self._ops = dict()
        self.verbose = verbose
        self.stdout = stdout
        self.log = lib.logger.new_logger(self, verbose=verbose)
        self.norm_b = norm_b
        self.norm_k = norm_k

    # def calc_op(self, opname):
    def set_op(self, opname:str, array:np.ndarray) -> None:
        self._ops[opname] = array
    
    def get_op_normalized(self, opname:str) -> None:
        return self.get_op(opname, normalized=False) / np.einsum("i,j->ij", self.norm_b, self.norm_k)
    
    def get_op(self, opname, normalized=False):
        if not normalized:
            op = self._ops[opname]
            return op
        else:
            return self.get_op_normalized(opname)
    
    def dump(self, scratch, tag):
        np.save('%s/%s-OPS.npy' % (scratch, tag), self._ops, allow_pickle=True)

    def load(self, scratch, tag):
        self._ops = np.load('%s/%s-OPS.npy' % (scratch, tag), allow_pickle=True).item()

    def transpose(self):
        cd_new = ContractData(self.norm_k, self.norm_b, 
                              self.verbose, self.stdout)
        op_names = self._ops.keys()
        for op_name in op_names:
            cd_new.set_op(op_name, self._ops[op_name].T)
        return cd_new

    def __add__(self, other):
        cd_new = ContractData(self.norm_k, self.norm_b, 
                              self.verbose, self.stdout)
        op_names = self._ops.keys()
        for op_name in op_names:
            if isinstance(other, ContractData):
                cd_new.set_op(op_name, self._ops[op_name] + other._ops[op_name])
            else:
                cd_new.set_op(op_name, self._ops[op_name] + other)
        return cd_new

    def __mul__(self, scalar):
        cd_new = ContractData(self.norm_k, self.norm_b, 
                              self.verbose, self.stdout)
        op_names = self._ops.keys()
        for op_name in op_names:
            cd_new.set_op(op_name, self._ops[op_name] * scalar)
        return cd_new

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self.__mul__(1. / scalar)

    @property
    def T(self): return self.transpose()


def build_excitation_space(
    tag_gs:str,
    impidxs:Iterable,
    operators,
    tag_ts,
    norb,
    nelec,
    restart=True,
    verbose=lib.logger.INFO,
    scratch_cd="./cd",
    scratch_ts="./tempdir",
    calc_eom=False,
    ham=None,
    **kwargs
):

    if scratch_ts[-1] == '/':
        scratch_ts = scratch_ts[:-1]
    if scratch_cd[-1] == '/':
        scratch_cd = scratch_cd[:-1]

    ket = np.load('%s/%s.npy' % (scratch_ts, tag_gs))

    norm_path = '%s/%s-NORM.npy' % (scratch_cd, tag_ts)

    if restart:
        if not os.path.exists(norm_path):
            restart = False
    if not restart:

        log = lib.logger.new_logger(verbose=verbose)
        
        if calc_eom: ket_res = _Hc(ham, ket, norb, nelec)
        else:
            ket_res = ket
        if calc_eom:
            log.debug("\t\t In build_excitation_space: Saving the P_i H |gs> to tag %s" % tag_ts)
        build_patch_vectors_from_ops(ket_res, impidxs, operators,
                                     norb, nelec, scratch_ts, tag_ts, log)
        dim = len(operators)
        norm = calc_norm(norb, nelec, scratch_ts, tag_ts, dim, log)
        np.save(norm_path, norm)
    else:
        norm = np.load(norm_path)
    return norm


def get_contract_data(tag_cd:str, op_name:str, op:tuple,
                      tss_bra:LocalExcitationBases,
                      tss_ket:LocalExcitationBases,
                      restart=True, calc_eom=True,
                      norm_bra_given=None, norm_ket_given=None,
                      ):
    scratch_cd = tss_ket.scratch_cd

    if scratch_cd[-1] == '/':
        scratch_cd = scratch_cd[:-1]

    norm_b = tss_bra.run(restart=restart)
    norm_k = tss_ket.run(restart=restart)

    if norm_bra_given is not None:
        norm_b = norm_bra_given
    if norm_ket_given is not None:
        norm_k = norm_ket_given

    if calc_eom:
        tss_bra.run(restart=restart, calc_eom=True, ham=op)
        tss_ket.run(restart=restart, calc_eom=True, ham=op)

    dim_bra, dim_ket = len(norm_b), len(norm_k)

    cd = ContractData(norm_b, norm_k, tss_ket.verbose, tss_ket.stdout)
    op_data_path = '%s/%s-OPS.npy' % (scratch_cd, tag_cd)
    if os.path.exists(op_data_path):
        cd.load(scratch_cd, tag_cd)

    if op_name not in cd._ops.keys():
        arr = tss_ket.contract_with_bra(op=op,
            tag_bra=tss_bra.tag_ts, tag_ket=tss_ket.tag_ts,
            dim_bra=dim_bra, dim_ket=dim_ket)
        if calc_eom:
            arr -= .5 * tss_ket.contract_with_bra(op=None,
                    tag_bra=tss_bra.tag_ts_eom, tag_ket=tss_ket.tag_ts,
                    dim_bra=dim_bra, dim_ket=dim_ket)

            arr -= .5 * tss_ket.contract_with_bra(op=None,
                    tag_bra=tss_bra.tag_ts, tag_ket=tss_ket.tag_ts_eom,
                    dim_bra=dim_bra, dim_ket=dim_ket)

        cd.set_op(op_name, arr)
        cd.dump(scratch_cd, tag_cd)
    return cd


def run_local_excitation_per_task(mydmet, imp_id:int, task:Iterable, opname:str, calc_eom:bool) -> ContractData:
    """
    Contract for ovlp or ham_dmet from a given task.

    imp_id: the index of the impurity

    task: Supported task types:
        len(task) == 6: task_type, bra_rel, ket_rel, bra_head, ket_head, k
        If task_type == 'IJI', the whole impurity I+J are participated to calculate ham_dmet;
        If task_type == 'IJK', only the looped indices K evaluates ham_dmet.

            op_name: 'H' for Hamiltonian, 'S' for overlap

    calc_eom: Whether we approximate from [H,P]|Psi> = (E-E_0)P|Psi> or HP|Psi> = EP|Psi>
    """
    hcore_lo, veff_lo, rdm1_lo, scratch_ts, scratch_cd, op_scheme, verbose = map(mydmet.__dict__.get,
            ('hcore_lo', 'veff_lo', 'rdm1_lo', 'scratch_ts', 'scratch_cd', 'op_scheme', 'verbose'))
    imp_idx, C_lo_eo, h2_emb = map(mydmet.impurities[imp_id].get, ('imp_idx', 'C_lo_eo', 'h2_emb'))

    mat_type, braidx_rel, ketidx_rel = task[:3]
    tag_cd = "%s-%d-%d-%d" % (mat_type, *task[-3:])
    tag_gs = "GS-%d" % imp_id
    tag_bra = tag_cd + "-bra"
    tag_ket = tag_cd + "-ket"

    C_lo_eo = get_bath(rdm1_lo, imp_idx)
    neo = C_lo_eo.shape[1]
    rdm1_emb = C_lo_eo.T @ mydmet.rdm1_lo @ C_lo_eo

    if mat_type == 'IJI':
        impidx_in_emb = list(range(len(imp_idx)))
    elif mat_type in 'IJK':
        if len(task) == 6:
            # traditional (type, brarel, ketrel, Iabs, Jabs, Kabs)
            idx_k_abs = task[-1]
            idx_k_rel = imp_idx.index(idx_k_abs)
            impidx_in_emb = [idx_k_rel]
        elif len(task) == 7:
            # for bunch DP (type, brarel, ketrel, dprel, Iabs, Jabs, Kabs)
            impidx_in_emb = task[3]
        else:
            raise NotImplementedError('Unknown task type!')

    if opname == 'H':       # Hamiltonian matrix
        ham = get_ham_dmet(hcore_lo, veff_lo, h2_emb, rdm1_emb, impidx_in_emb, C_lo_eo)

    elif opname == 'S':     # Overlap matrix
        ham = None

    bra_kwargs = {'impidxs': braidx_rel, 'tag_ts': tag_bra}
    ket_kwargs = {'impidxs': ketidx_rel, 'tag_ts': tag_ket}

    common_kwargs = {
        'norb': neo,
        'nelec': _unpack_nelec(neo),
        'restart': True,
        'op_scheme': op_scheme,
        'operators': [],
        'tag_gs': tag_gs,
        'scratch_ts': scratch_ts,
        'scratch_cd': scratch_cd}

    bra_kwargs.update(common_kwargs)
    ket_kwargs.update(common_kwargs)

    tss_bra = LocalExcitationBases(verbose=verbose, stdout=sys.stdout, **bra_kwargs)
    tss_ket = LocalExcitationBases(verbose=verbose, stdout=sys.stdout, **ket_kwargs)

    cd = get_contract_data(
        tag_cd=tag_cd, op_name=opname, op=ham,
        tss_bra=tss_bra, tss_ket=tss_ket,
        restart=True, calc_eom=calc_eom,
    )
    return cd


def collect_data_per_task(scratch_cd, task, **kwargs):

    if scratch_cd[-1] == '/':
        scratch_cd = scratch_cd[:-1]

    task_type, bra_rel, ket_rel = task[:3]
    tag_cd = "%s-%d-%d-%d" % (task_type, *task[-3:])
    tag_bra = tag_cd+"-bra"
    tag_ket = tag_cd+"-ket"
    norm_bra_path = "%s/%s-NORM.npy" % (scratch_cd, tag_bra)
    norm_ket_path = "%s/%s-NORM.npy" % (scratch_cd, tag_ket)
    norm_bra = np.load(norm_bra_path)
    norm_ket = np.load(norm_ket_path)
    cd = ContractData(norm_bra, norm_ket, **kwargs)
    cd.load(scratch_cd, tag_cd)
    return cd


def collect_data(taskss:dict, scratch:str, **kwargs):
    """
    eompairs: (idx_bra, idx_ket)
    """
    log = lib.logger.new_logger(verbose=kwargs.get('verbose', 4))

    eominfos = dict()

    for itasks, (frag_idx, tasks) in enumerate([(k, v) for k, v in taskss.items()]):
        for task in tasks:
            cd = collect_data_per_task(scratch, task, **kwargs)
            task_type = task[0]
            id1, id2, id3 = task[-3:]
            if task_type[-1] == 'I':
                log.debug("Collect Effective Hamiltonian data "
                          "<Psi_0| P(%-2d) H P(%-2d) | Psi_0>", id1, id2)
                eominfos[(id1, id2)] = cd
            elif task_type[-1] == 'K':
                log.debug("Collect Effective Hamiltonian data "
                    "<Psi_0| P(%-2d) H(%-2d) P(%-2d) | Psi_0>", id1, id3, id2)
                eominfos[(id1, id2, id3)] = cd
    return eominfos


def _search_data(latt, eominfos, symmap_key):
    symmap_val = latt.sym_map[symmap_key]
    real_key, tp = symmap_val[:-1], symmap_val[-1]
    if tp == 1:
        data = eominfos.get(real_key, None)
    else:
        data = eominfos[real_key].T
    return data


def get_op_blocks(latt, eominfos, opname, normalized=True):
    """
    From the eominfos ready to export observables, get the Hamiltonian/operator matrix for each independent block.
    """
    op_dict = dict()

    eompairs = latt.eom_pairs
    symmap = latt.sym_map
    N = latt.L
    for i, (braidx, ketidx) in enumerate(eompairs):
        op_dict[(braidx, ketidx)] = 0.
        cd = _search_data(latt, eominfos, (braidx, ketidx))
        if cd is not None:
            op_dict[(braidx, ketidx)] += cd.get_op(opname, normalized)
        if opname != "S":
            for k in range(N):
                if (braidx, ketidx, k) in symmap.keys():
                    cd = _search_data(latt, eominfos, (braidx, ketidx, k))
                    if cd is not None:
                        res = cd.get_op(opname, normalized)
                        op_dict[(braidx, ketidx)] += res
    return op_dict


def load_ops_from_file(latt:ChainLattice, taskss, scratch, op_names=None, normalized=True):
    """
    get all data from all tasks and stored files.
    
    latt: ChainLattice object
    op_names: if None, detect from files
    normalized: whether the operators are normalized (diag of S == 1)
    """
    eominfos = collect_data(taskss, scratch, verbose=latt.verbose, stdout=latt.stdout)
    if op_names is None:
        op_names = list(eominfos[(0, 0)]._ops.keys())
    for op_name in op_names:
        latt.eom_data[op_name] = get_op_blocks(latt, eominfos, op_name, normalized=normalized)


class DMET4Excitation:
    def __init__(self, hcore_lo, veff_lo, eri_lo, rdm1_lo, lattice:ChainLattice,
                 scratch_ts, scratch_cd, op_scheme,
                 calc_eom=True, verbose=4, stdout=sys.stdout):
        """
        Main driver for DMET for calculating the excitation properties.

        hcore_lo, veff_lo, eri_lo, rdm1_lo: lattice Hamiltonians and density matrices dumped from mean-field results
        lattice: ChainLattice object, which contains the impurity indices and tasks

        scratch_ts: scratch directory for local excitation vectors
        scratch_cd: scratch directory for data of effective Hamiltonian part.
        op_scheme: operator schemes for local excitations.
        calc_eom: Whether we approximate from [H,P]|Psi> = (E-E_0)P|Psi> or HP|Psi> = EP|Psi>
            NOTE: "eom" named from the quantum chemistry method "EOM-CC" (equation-of-motion coupled cluster),
            which use the similar treatment to directly calculate the excitation energy.
            This project always use calc_eom=True.
        """
        self.verbose = verbose
        self.stdout = stdout
        self.log = lib.logger.new_logger(self, verbose=verbose)
        self.lattice = lattice
        self.hcore_lo = hcore_lo
        self.veff_lo = veff_lo
        self.eri_lo = eri_lo
        self.rdm1_lo = rdm1_lo
        self.scratch_ts = scratch_ts
        self.scratch_cd = scratch_cd
        self.op_scheme = op_scheme
        self.calc_eom = calc_eom

        self.impurities = [dict(imp_idx=imp_idx, tasks=tasks)
            for _, (imp_idx, tasks) in enumerate(zip(lattice.tasks.keys(), lattice.tasks.values()))]

        os.makedirs(scratch_ts, exist_ok=True)
        os.makedirs(scratch_cd, exist_ok=True)

    def solve_ground_states(self):
        self.log.info("Running ground-state calculations.")
        for imp_id, imp in enumerate(self.impurities):
            self.log.debug("Solving ground state for impurity %d ," % imp_id \
                          + " Impurity indices: %s", imp['imp_idx'])
            C_lo_eo = get_bath(self.rdm1_lo, imp['imp_idx'])
            neo = C_lo_eo.shape[1]
            h2_emb = get_h2_emb(self.eri_lo, C_lo_eo)
            h1_emb = get_h1_emb(self.hcore_lo, self.rdm1_lo, C_lo_eo, h2_emb)
            fci_solver = fci.direct_spin1.FCIBase()
            eci, ci = fci_solver.kernel(h1_emb, h2_emb, neo, neo)
            np.save("%s/GS-%d.npy" % (self.scratch_ts, imp_id), ci)
            self.impurities[imp_id].update(dict(C_lo_eo=C_lo_eo, neo=neo, h1_emb=h1_emb, h2_emb=h2_emb))
        return self

    def run_local_excitations(self):
        self.log.info("Running local excitation calculations ...")
        for imp_id, imp in enumerate(self.impurities):
            for task in imp['tasks']:
                for op_name in ['H', 'HS'][task[-3] == task[-1]]:
                    run_local_excitation_per_task(self, imp_id, task, op_name,
                        calc_eom=self.calc_eom and (op_name == "H"))
        return self

    def collect_effective_hamiltonian(self, normalized=True):
        self.log.info("Collecting effective Hamiltonians ...")
        load_ops_from_file(self.lattice, self.lattice.tasks,
                           self.scratch_cd, op_names=["H", "S"],
                           normalized=normalized)
        return self

    def kernel(self, **kwargs):
        self.solve_ground_states().run_local_excitations().collect_effective_hamiltonian()
        return self.lattice.eig(**kwargs)

    def _finalize(self):
        os.remove(self.scratch_ts)
        os.remove(self.scratch_cd)
