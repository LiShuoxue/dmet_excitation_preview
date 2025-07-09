"""
chain_lattice.py
"""

import sys, os
import numpy as np
from scipy import linalg as la
from itertools import chain
from collections.abc import Iterable
from functools import partial
from pyscf import lib


def gen_uniform_grid(L):
    # [0 1 2 3 ...]
    return [list(range(L))]


def check_grid_kind(coord, grids):
    for i, grid in enumerate(grids):
        for g in grid:
            if coord == g: return i
    return None


def coord_center_to_idxs(coord, extend_idx_rule, L, ns=None):
    # extend_idx_rule: list like (-1, 0, 1, 2)
    if not isinstance(extend_idx_rule, Iterable):
        if extend_idx_rule is None:
            extend_idx_rule = range(ns)
        else:
            extend_idx_rule = list(range(extend_idx_rule))
    coords = [(coord+x)%L for x in extend_idx_rule]
    return coords


def translate_coords(coords, grids, L, idx=0):
    coord_center = coords[idx]
    grid_kind_center = check_grid_kind(coord_center, grids)
    assert grid_kind_center is not None
    coord_target = grids[grid_kind_center][0]
    dx = coord_target - coord_center
    res = list()
    for coord in coords:
        res.append((coord+dx)%L)
    return tuple(res)


translate_coords_uniform = lambda coords, L, idx=0: translate_coords(
        coords, gen_uniform_grid(L), L, idx
    )


def get_equivalent_block_IJ(L, translate_coord_func):
    """
    return a dictionary:
        key
        val
    """
    uniform_grid = gen_uniform_grid(L)[0]
    equivalent_coords = list()
    
    for coord_I in uniform_grid:
        for coord_J in uniform_grid:
            coords = (coord_I, coord_J)
            coords_aligned_0 = translate_coord_func(coords, L, idx=0)
            coords_aligned_1 = translate_coord_func(coords, L, idx=1)
            equivalent_coords.append((coords_aligned_0, coords_aligned_1, coords))
    
    visited = list()
    eq_cls_full = dict()
    for c0, c1, c in equivalent_coords:
        if c0 not in visited:
            if c0 not in eq_cls_full.keys():
                eq_cls_full[c0] = list()

        # Shuoxue NOTE: last index denotes whether transportation is needed. 
        # (1 for nothing, -1 for transpose)
        images = [(c1[0], c1[1], 1), (c[0], c[1], 1, ), (c1[1], c1[0], -1), (c[1], c[0], -1)]

        for im in images:
            if c0 in eq_cls_full.keys():
                if im not in eq_cls_full[c0]:
                    eq_cls_full[c0].append(im)
                    visited.append(im[:2])
    return eq_cls_full


def get_equivalent_block_IJk(L, ns, translate_coord_func):
    uniform_grid = gen_uniform_grid(L)[0]
    equivalent_coords = list()
    
    for coord_I in uniform_grid:
        coord_I_full = tuple([(coord_I+x)%L for x in range(ns)])
        for coord_J in uniform_grid:
            coord_J_full = tuple([(coord_J+x)%L for x in range(ns)])
            for coord_k in uniform_grid:
                if coord_k not in coord_I_full + coord_J_full:
                    coords = (coord_I, coord_J, coord_k)
                    coords_aligned_0 = translate_coord_func(coords, L, idx=0)
                    coords_aligned_1 = translate_coord_func(coords, L, idx=1)
                    equivalent_coords.append((coords_aligned_0, coords_aligned_1, coords))

    visited = list()
    eq_cls_full = dict()
    for c0, c1, c in equivalent_coords:
        if c0 not in visited:
            if c0 not in eq_cls_full.keys():
                eq_cls_full[c0] = list()

        images = [c1 + (1, ), c + (1, ), (c1[1], c1[0], c1[2], -1), (c[1], c[0], c[2], -1)]
        # if c0 not in visited
        for im in images:
            if c0 in eq_cls_full.keys():
                if im not in eq_cls_full[c0]:
                    eq_cls_full[c0].append(im)
                    visited.append(im[:3])
            
    return eq_cls_full


get_equivalent_block_IJ_uniform = lambda L: get_equivalent_block_IJ(
    L, translate_coords_uniform
)
get_equivalent_block_IJk_uniform = lambda L, ns: get_equivalent_block_IJk(
    L, ns, translate_coords_uniform
)


def get_symmap(L, ns, eq_cls_IJ, eq_cls_IJK):
    """
    Get the symmetry map from the equivalent classes of blocks.
    """
    # if eq_cls_IJ is No
    sym_map = dict()
    for idx_r in eq_cls_IJ.keys():
        for eq_coord in eq_cls_IJ[idx_r]:
            idx_l = eq_coord[:2]
            if idx_l not in sym_map.keys():
                sgn = eq_coord[-1]
                sym_map[idx_l] = idx_r + (sgn, )

    for idx_r in eq_cls_IJK.keys():
        for eq_coord in eq_cls_IJK[idx_r]:
            idx_l = eq_coord[:3]
            if idx_l not in sym_map.keys():
                sgn = eq_coord[-1]
                sym_map[idx_l] = idx_r + (sgn, )
    
    return sym_map


def get_tasks(L, ns, eq_cls_IJ, eq_cls_IJK, log=None, extend_idx_rule=None):
    """
    Get the indepenent tasks of DMET excitation.
    """
    if extend_idx_rule is None:
        extend_idx_rule = list(range(ns))

    idx_full_nonred_IJ = [
        tuple(sorted(set(chain(
            *[coord_center_to_idxs(coord, extend_idx_rule, L) for coord in x]
        )))) for x in eq_cls_IJ.keys()
    ]

    idx_full_nonred_IJk = [
        tuple(sorted(set(chain(
            *[
                coord_center_to_idxs(x[0], extend_idx_rule, L),
                coord_center_to_idxs(x[1], extend_idx_rule, L),
                [x[2]]
            ]
        )))) for x in eq_cls_IJK.keys()
    ]
    
    imps = list(set(idx_full_nonred_IJ + idx_full_nonred_IJk))
    tasks = dict()
    for imp in imps: tasks[imp] = list()

    for i, coords_abs in enumerate(eq_cls_IJ.keys()):
        imp = idx_full_nonred_IJ[i]
        
        idx_abs = [coord_center_to_idxs(coord, list(range(ns)), L)
                           for coord in coords_abs]
        coords_rel = [tuple([imp.index(x) for x in y]) for y in idx_abs]
        task = ('IJI', coords_rel[0], coords_rel[1],
                coords_abs[0], coords_abs[1], coords_abs[0])

        tasks[imp].append(task)

    for i, coords_abs in enumerate(eq_cls_IJK.keys()):
        imp = idx_full_nonred_IJk[i]

        idx_abs = [coord_center_to_idxs(coord, list(range(ns)), L)
                           for coord in coords_abs[:2]]
        idx_abs_full = [coord_center_to_idxs(coord, extend_idx_rule, L)
                        for coord in coords_abs[:2]]
        coords_rel = [tuple([imp.index(x) for x in y]) for y in idx_abs]

        if coords_abs[2] not in idx_abs_full[0] + idx_abs_full[1]:
            task = ('IJK', coords_rel[0], coords_rel[1],
                    coords_abs[0], coords_abs[1], coords_abs[2])
            tasks[imp].append(task)
    return tasks


def get_eom_pairs_uniform(L):
    return [(0,x) for x in range(L)]


def stripe2full(op_dict, L, translate_func):
    dim_block = op_dict[(0,0)].shape[0]

    op_full = np.zeros((dim_block*L, dim_block*L))
    for ili in range(L):
        for ilj in range(L):
            coords = (ili, ilj)
            coords_aligned = translate_func(coords, L, idx=0)
            
            mesh = np.ix_(range(dim_block*ili, dim_block*(ili+1)),
                          range(dim_block*ilj, dim_block*(ilj+1)))
            op_full[mesh] += op_dict[coords_aligned]
    return op_full


stripe2full_uniform = lambda op_dict, L : stripe2full(op_dict, L, translate_coords_uniform)


def opdict2stripe(op_dict, L):
    """given an operator dict, return the stripe representation."""
    dimx, dimy = op_dict[(0, 0)].shape
    afm = False
    for key in op_dict.keys():
        if key[0] == 1:
            afm = True
            break
    if afm:
        assert ~(L//2)
        res = np.zeros((L//2, dimx*2, dimy*2))
        for i in range(L//2):
            res[np.ix_([i], range(dimx), range(dimy))] += op_dict[(0, 2*i)]
            res[np.ix_([i], range(dimx), range(dimy, 2*dimy))] += op_dict[(0, 2*i+1)]
            res[np.ix_([i], range(dimx, 2*dimx), range(dimy))] += op_dict[(1, 2*i)]
            res[np.ix_([i], range(dimx, 2*dimx), range(dimy, 2*dimy))] += op_dict[(1, 2*i+1)]
    else:
        res = np.zeros((L, dimx, dimy))
        for i in range(L):
            res[i] += op_dict[(0, i)]
    return res


def init_lattice(latt, **kwargs):
    latt.opdict2stripe = partial(opdict2stripe, L=latt.L)
    latt.eq_cls_IJ   = get_equivalent_block_IJ_uniform(latt.L)
    latt.eq_cls_IJk  = get_equivalent_block_IJk_uniform(latt.L, latt.ns)
    latt.eom_pairs   = get_eom_pairs_uniform(latt.L)

    latt.get_imp_by_center = lambda coord: coord_center_to_idxs(
    coord, latt.extend_idx_rule, latt.L, latt.ns)
    latt.sym_map     = get_symmap(latt.L, latt.ns,
                                latt.eq_cls_IJ, latt.eq_cls_IJk)
    latt.tasks       = get_tasks(latt.L, latt.ns,
                                latt.eq_cls_IJ, latt.eq_cls_IJk,
                                extend_idx_rule=latt.extend_idx_rule)
    latt.kmesh = np.array([latt.L, 1, 1])


def check_task(topo):
    # Shuoxue TODO: support screening
    taskss = topo.tasks
    coords_cls_IJ = topo.eq_cls_IJ.keys()
    coords_cls_IJk = topo.eq_cls_IJk.keys()

    task_IJ_list = list()
    task_IJk_list = list()
    for imp_idx, tasks in taskss.items():
        for task in tasks:
            label, relidxs_bra, relidxs_ket, \
            absidx_bra_head, absidx_ket_head, dploop_idx = task
            if label == 'IJI':
                task_IJ_list.append([topo.idx2coord(x) for x in task[3:5]])
            elif label == 'IJK':
                task_IJk_list.append([topo.idx2coord(x) for x in task[3:]])
            else:
                raise ValueError('Type (%s) other than IJI and IJK '
                                 'are in the task list!' % label)
            if imp_idx[relidxs_bra[0]] != absidx_bra_head:
                raise ValueError('relative index of bra %d from '
                                 'imp_idx %d != absolute index %d !' % (
                    relidxs_bra[0], imp_idx, absidx_bra_head))
            elif imp_idx[relidxs_ket[0]] != absidx_ket_head:
                raise ValueError('relative index of ket %d from '
                                 'imp_idx %d != absolute index %d !' % (
                    relidxs_ket[0], imp_idx, absidx_ket_head))

    # check the completeness of IJI tasks
    coords_cls_IJ = set([tuple(x) for x in coords_cls_IJ])
    task_IJ_list = set([tuple(x) for x in task_IJ_list])
    assert coords_cls_IJ == task_IJ_list

    # check the completeness of IJk tasks
    coords_cls_IJk_with_extend = list()
    for coord_cls_IJk in coords_cls_IJk:
        bra_head, ket_head, dp_coord = coord_cls_IJk
        if dp_coord not in [topo.idx2coord(x) for x in topo.get_imp_by_center(bra_head) + topo.get_imp_by_center(ket_head)]:
            coords_cls_IJk_with_extend.append(coord_cls_IJk)

    coords_cls_IJk = set([tuple(x) for x in coords_cls_IJk_with_extend])
    task_IJk_list = set([tuple(x) for x in task_IJk_list])
    assert coords_cls_IJk == task_IJk_list


def dump_info(topo):
    log = topo.log

    log.info('-' * 20 + 'Lattice information' + '-' * 20)
    if topo.dimension == 1:
        log.info('L               = %d' % topo.L)
        log.info('ns              = %d' % topo.ns)
    elif topo.dimension == 2:
        log.info('(lx, ly)        = %d %d' % (topo.lx, topo.ly))
        log.info('coord_1cell     = {}'.format(topo.coord_1cell))
    else:
        raise ValueError('Unknown dimension %d.' % topo.dimension)
    log.info('extend_idx_rule = {}'.format(topo.extend_idx_rule))

    log.debug('-' * 20 + 'Full list of tasks' + '-' * 20)
    for idx, task in enumerate(topo.tasks.items()):
        log.debug('{} : {}'.format(idx, task))

    log.debug('-' * 20 + 'Independent blocks' + '-' * 20)
    for eom_pair in topo.eom_pairs:
        log.debug(f'{eom_pair}')

    log.debug1('-' * 20 + 'Full list of symmetry map' + '-' * 20)
    for key, val in topo.sym_map.items():
        log.debug1('{} : {}'.format(key, val))


def eigh(H, S, thres=None, mask=None, thres_min=None):
    wS, vS = la.eigh((S+S.T.conj())/2)
    
    if mask is None:
        if thres is None:
            thres = np.abs(min(wS))
            if thres_min is not None:
                thres = max(thres, thres_min)
        mask = wS > thres
    S_proj = vS.T.conj()[mask] @ (S+S.T.conj())/2 @ vS[:,mask]
    H_proj = vS.T.conj()[mask] @ (H+H.T.conj())/2 @ vS[:,mask]
    wHp, vHp = la.eigh(H_proj, S_proj)
    return wHp, vHp, wS, vS, mask

def fft2k(A, kmesh, axis=0):
    """
    axis : where the R/k label is.
    """
    shape_new = A.shape[:axis] + tuple(kmesh) + A.shape[axis+1:]
    axes_fft = range(axis, axis + len(kmesh))
    return np.fft.rfftn(a=A.reshape(shape_new), axes=axes_fft).reshape(A.shape)


def R2k(dm_R, kmesh):
    return fft2k(dm_R, kmesh, axis=dm_R.ndim - 3)


def simple_eig(topo, ham_name='H', ovlp_name='S', thres=None, masks=None, matrix_mask=None, thres_min=None):
    """
    thres, mask: mask the Hamiltonian and ovlp via the eigenvalue of ovlp matrix.
    matrix_mask: which part of the operator matrix is used.
    """
    log = topo.log

    op_dict_H = topo.eom_data[ham_name]
    op_dict_S = topo.eom_data[ovlp_name]
    H_R = topo.opdict2stripe(op_dict_H)
    S_R = topo.opdict2stripe(op_dict_S)
    H_k = R2k(H_R, topo.kmesh)
    S_k = R2k(S_R, topo.kmesh)
    nkpts = topo.L

    eig_results = list()
    eigv_results = list()

    for k in range(nkpts):
        kpt = k
        thre, mask, thre_min = None, None, None
        if isinstance(thres, Iterable):
            thre = thres[k]
        else:
            thre = thres
        if isinstance(masks, Iterable):
            mask = masks[k]
        else:
            mask = masks
        if isinstance(thres_min, Iterable):
            thre_min = thres_min[k]
        else:
            thre_min = thres_min

        if matrix_mask is not None:
            H = H_k[k][np.ix_(matrix_mask, matrix_mask)]
            S = S_k[k][np.ix_(matrix_mask, matrix_mask)]
        else:
            H, S = H_k[k], S_k[k]

        wHp, vHp, wS, vS, mask = eigh(H, S,
                            thres=thre, mask=mask, thres_min=thre_min)
        log.debug('k-point {}, Singular values of ovlp \n {}'.format(kpt, wS))
        log.debug('Energy eigenvalue = \n {}'.format(wHp))
        eig_results.append(wHp)
        eigv_results.append(np.dot(vS[:,mask], vHp))

    return eig_results, eigv_results


class ChainLattice(object):
    """Lattice object for DMET excitation project"""
    def __init__(self, verbose=5, **kwargs):
        # system info

        self.dimension = 1
        self.L = None
        self.ns = None
        self.kmesh = None
        self.coord_1cell = None

        # dump information
        self.verbose = verbose
        self.stdout = sys.stdout
        self.log = lib.logger.new_logger(self, verbose=self.verbose)

        # properties
        self.eq_cls_IJ = {}
        self.eq_cls_IJk = {}
        self.extend_idx_rule = None
        self.sym_map = {}
        self.tasks = {}
        self.eom_pairs = []

        # collection of excitation solutions
        # keys: operator names
        self.eom_data = {}

        # convenience functions
        self.get_imp_by_center = lambda coord: ...
        self.opdict2stripe = lambda op_dict: ...
        self.coord2idx = lambda coord: coord
        self.idx2coord = lambda idx: idx

        # initialize system info
        args_info = ('L', 'ns', 'coord_1cell')
        input_kwargs = {'extend_idx_rule': None}
        input_kwargs.update({key : val for key, val in kwargs.items() 
                        if key in args_info})
        self.__dict__.update(**input_kwargs)

        # initialize information of topology
        init_lattice(self)
        # check whether tasks are correctly generated
        self.check_task()
        # output the information of the topology generator
        self.dump_info()

    # functions related to getting inner properties
    check_task = check_task
    dump_info  = dump_info
    eig = simple_eig
