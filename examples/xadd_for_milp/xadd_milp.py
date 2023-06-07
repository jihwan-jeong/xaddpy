import json
import os.path as path
from typing import List, Tuple

import sympy as sp

from xaddpy.utils.logger import logger
from xaddpy.utils.util import compute_rref_filter_eq_constr
from xaddpy.xadd.xadd import XADD as _XADD


class XADD(_XADD):
    def __init__(self, args: dict = ...):
        super().__init__(args)
        self._config_json_fname = None
        self._num_decision_vars = 0
        self._feature_dim = None        # for EMSPO
        self._param_dim: int = None
        
    def link_json_file(self, config_json_fname: str):
        self._config_json_fname = config_json_fname
    
    def set_feature_dim(self, dim: int):
        self._feature_dim = dim
    
    def set_param_dim(self, dim):
        self._param_dim = dim
    
    def update_decision_vars(self, min_var_set, free_var_set):
        """Note: does not support binary variables"""
        variables = min_var_set.union(free_var_set)
        self._decisionVars = variables
        self._num_decision_vars = len(variables)
        self._min_var_set.update(min_var_set)
        self._free_var_set.update(free_var_set)
    
    """
    Export and import XADDs for MILP problems.
    """
    def export_min_or_max_xadd(self, min_or_max_node_id: int, fname: str):
        """
        Export the min xadd to a file named `fname`.
        Some additional information is also exported, for example:
            
        """
        with open(fname, 'w+') as f:
            # Write information regarding problem definition
            f.write(f'Config file: {self._config_json_fname}\n')
            dec_vars = ', '.join(
                [str(v) for v in sorted(self._free_var_set, key=lambda x: str(x))]
            )
            f.write(f'Decision variables: {dec_vars}\n')
            bounds = ':'.join(
                [f'{bound}' for v, bound in sorted(self._var_to_bound.items(),
                                                   key=lambda x: str(x[0]))
                            if v in self._free_var_set]
            )
            f.write(f'Bounds: {bounds}\n')

            # Write the LP
            f.write(f'\nlp: ')
        self.export_xadd(min_or_max_node_id, fname, append=True)

    def export_argmin_or_max_xadd(self, fname: str):
        """
        Export the argmin xadd of `min_var` variables to a file named `fname`.
        Some additional information is also exported: such as
            name of decision variables
            bounds of those variables
            dimensionality of parameters and features

        :param fname:       (str) the file name to which argmin(max) xadd is exported
        """
        # Write information regarding problem definition
        with open(fname, 'w+') as f:
            f.write(f'Config file: {self._config_json_fname}\n')
            dec_vars = ', '.join(
                [str(v) for v in sorted(self._min_var_set, key=lambda x: str(x))]
            )
            f.write(f'Decision variables: {dec_vars}\n')
            bounds = ':'.join(
                [f'{bound}' for bound in sorted([b for v, b in self._var_to_bound.items()
                                                 if v in self._min_var_set],
                                                key=lambda x: str(x))]
            )
            f.write(f'Bounds: {bounds}\n')
            
            # For EMSPO
            if self._feature_dim is not None:
                f.write('Feature dimension: {}\n'.format(self._feature_dim))

        for var in self._min_var_set:
            with open(fname, 'a+') as f:
                f.write(f'\n{str(var)}: ')
            var_arg_min_or_max = self._var_to_anno.get(var)
            self.export_xadd(var_arg_min_or_max, fname, append=True)

    def import_lp_xadd(self, fname) -> Tuple[list, dict]:
        """
        Read in the XADD corresponding to an LP.
        """
        ns = {}
        xadd_str = ''
        dec_var = None

        # Check existence of the file
        if not path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist, raising an error")
        
        with open(fname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line_split = line.split(':')
                if i == 0:
                    assert line_split[0].lower() == 'config file', "Path to the configuration file should be provided"
                    json_file = fname.replace('.xadd', '.json')
                    logger.info(line.strip())
                    self.link_json_file(json_file)

                elif i == 1:
                    assert line_split[0].lower() == 'decision variables', "Decision variables should be specified in the file"
                    symbols = line_split[1].strip().split(',')
                    dec_vars = sp.symbols(symbols)
                    dec_vars = list(sorted(dec_vars,
                                           key=lambda x: (float(str(x).split("_")[0][1:]), float(str(x).split("_")[1]))
                                           if len(str(x).split("_")) > 1 else float(str(x)[1:])))
                    ns.update({str(v): v for v in dec_vars})
                    logger.info(line.strip())
                    self._free_var_set.update(dec_vars)

                elif i == 2:
                    assert line_split[0].lower() == 'bounds', "Bound information should be provided"
                    bound_dict = {ns[str(v)]: 
                                  tuple(map(float, line_split[i+1].strip()[1:-1].replace('oo', 'inf').split(',')))
                                  for i, v in enumerate(dec_vars)}
                    self._var_to_bound.update(bound_dict)
                    logger.info(line.strip())

                elif len(line_split) > 1:
                    if len(xadd_str) > 0:
                        raise ValueError
                    xadd_str = line_split[1].strip()
                    obj = sp.symbols(line_split[0])
                    ns.update({str(obj): obj})

                elif len(line.strip()) != 0:
                    xadd_str += line
            if len(xadd_str) > 0:
                lp = self.import_xadd(xadd_str=xadd_str, locals=ns)

        # Handle equality constraints if exist by looking at the configuration json file
        try:
            with open(self._config_json_fname, "r") as json_file:
                prob_instance = json.load(json_file)
        except:
            raise FileNotFoundError(f"Failed to open {self._config_json_fname}")

        eq_constr_dict, dec_vars = compute_rref_filter_eq_constr(prob_instance['eq-constr'],
                                                                 dec_vars,
                                                                 locals=ns)
        dec_vars = [v for v in dec_vars if v not in eq_constr_dict]

        # LP objective node
        self.set_objective(lp)

        return dec_vars, eq_constr_dict

    def import_arg_xadd(
            self, fname, feature_dim=None, param_dim=None, model_name=None
    ) -> Tuple[list, dict]:
        """
        Read in XADDs corresponding to argmin solutions.
        """
        ns = {}
        xadd_str = ''
        dec_var = None

        # Check existence of the file
        if not path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist")

        with open(fname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line_split = line.split(':')
                if i == 0:
                    assert line_split[0].lower() == 'config file', "Path to the configuration file should be provided"
                    json_file = fname.replace('_argmin.xadd', '.json')
                    self.link_json_file(json_file)

                elif i == 1:
                    assert line_split[0].lower() == 'decision variables', "Decision variables should be specified in the file"
                    dec_vars = sp.symbols(line_split[1].strip().replace(',', ' '))
                    dec_vars = list(sorted(dec_vars,
                                           key=lambda x: (float(str(x).split("_")[0][1:]), float(str(x).split("_")[1]))
                                           if len(str(x).split("_")) > 1 else float(str(x)[1:])))
                    ns.update({str(v): v for v in dec_vars})
                    self.update_name_space(ns)
                    self._min_var_set.update(dec_vars)
                    dim_dec_vars = len(self._min_var_set)

                elif i == 2:
                    assert line_split[0].lower() == 'bounds', "Bound information should be provided"
                    bound_dict = {ns[str(v)]: tuple(map(int, line_split[i+1].strip()[1:-1].split(','))) for i, v in enumerate(dec_vars)}
                    self._var_to_bound.update(bound_dict)

                elif i == 3 and line_split[0].lower() == 'feature dimension':
                    if feature_dim is None:
                        feature_dim = tuple(int(i) for i in line_split[1].strip().strip('(').strip(')').split(',') if i)
                    assert len(feature_dim) == 1

                    self.set_feature_dim(feature_dim)
                    if param_dim is None:
                        param_dim = (dim_dec_vars, feature_dim[0] + 1)
                    self.set_param_dim(param_dim)

                elif len(line_split) > 1:
                    if len(xadd_str) > 0 and dec_var is not None:
                        self._var_to_anno[dec_var] = self.import_xadd(xadd_str=xadd_str, locals=ns)
                        xadd_str = ''
                    xadd_str = line_split[1].strip()
                    dec_var = ns[line_split[0]]

                elif len(line.strip()) != 0:
                    xadd_str += line
            if len(xadd_str) > 0 and dec_var is not None:
                self._var_to_anno[dec_var] = self.import_xadd(xadd_str=xadd_str, locals=ns)
        # Handle equality constraints if exist by looking at the configuration json file
        try:
            with open(self._config_json_fname, "r") as json_file:
                prob_instance = json.load(json_file)
        except:
            raise FileNotFoundError(f"Failed to open {self._config_json_fname}")

        eq_constr_dict, dec_vars = compute_rref_filter_eq_constr(prob_instance['eq-constr'],
                                                                 dec_vars,
                                                                 locals=ns)
        dec_vars = [v for v in dec_vars if v not in eq_constr_dict]

        return dec_vars, eq_constr_dict

    def var_count(self, node_id, count=None):
        if count == None:
            count = dict()
        
        node = self.get_exist_node(node_id)
        expr = node.expr if node._is_leaf else self._id_to_expr.get(node.dec, None)

        if not node._is_leaf:
            self.var_count(node._low, count)
            self.var_count(node._high, count)

        for s in expr.free_symbols:
            if str(s).startswith('x'):
                count[s] = count.get(s, 0) + 1

        return count
