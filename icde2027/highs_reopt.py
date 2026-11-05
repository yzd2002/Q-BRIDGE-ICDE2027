"""Strong persistent-model LP comparator, with the same exact proof rule."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
import highspy

from .proof_check import CheckedView


class HighsReoptimization:
    def __init__(self,view,*,retain_basis=True):
        begin=time.perf_counter()
        self.checker=CheckedView(view);self.digest=self.checker.digest
        self.retain_basis=retain_basis;self.d=len(view['lower'])
        self.indices=np.arange(self.d,dtype=np.int32)
        self.h=highspy.Highs()
        options={'output_flag':False,'threads':1,'solver':'simplex',
                 'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10}
        for key,value in options.items():
            status=self.h.setOptionValue(key,value)
            if status!=highspy.HighsStatus.kOk:
                raise RuntimeError(f'HiGHS rejected registered option {key}')
        m=view['matrix'];j=len(view['rhs_lower'])
        lp=highspy.HighsLp();lp.num_col_=self.d;lp.num_row_=j
        lp.col_cost_=np.zeros(self.d);lp.col_lower_=np.array(view['lower']);lp.col_upper_=np.array(view['upper'])
        lp.row_lower_=np.array(view['rhs_lower']);lp.row_upper_=np.array(view['rhs_upper'])
        matrix=highspy.HighsSparseMatrix();matrix.num_col_=self.d;matrix.num_row_=j
        matrix.format_=highspy.MatrixFormat.kRowwise
        matrix.start_=np.asarray(m['indptr'],dtype=np.int32)
        matrix.index_=np.asarray(m['indices'],dtype=np.int32);matrix.value_=np.asarray(m['data'],dtype=float)
        lp.a_matrix_=matrix
        if self.h.passModel(lp)!=highspy.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected trusted view model')
        self.initialization_seconds=time.perf_counter()-begin
        self.native_version=self.h.version()

    def query(self,objective,threshold=.001):
        begin=time.perf_counter();c=np.asarray(objective,dtype=float)
        if c.shape!=(self.d,) or not np.all(np.isfinite(c)) or not np.isfinite(threshold):
            raise ValueError('finite matching objective and threshold required')
        basis_before=bool(self.h.getBasis().valid)
        solve_start=time.perf_counter()
        if not self.retain_basis:
            if self.h.clearSolver()!=highspy.HighsStatus.kOk:
                raise RuntimeError('Could not clear cold-comparator solver state')
        updated=self.h.changeColsCost(self.d,self.indices,c)
        basis_at_run=bool(self.h.getBasis().valid)
        run_status=self.h.run() if updated==highspy.HighsStatus.kOk else updated
        solve_seconds=time.perf_counter()-solve_start
        info=self.h.getInfo();solution=self.h.getSolution()
        common={'route':'highs_basis_hot_start' if self.retain_basis else 'highs_cold_solver',
                'screen_seconds':0.,'solve_seconds':solve_seconds,'bank_entries':0,
                'basis_valid_before_clear_or_update':basis_before,'basis_valid_at_run':basis_at_run,
                'basis_valid_after':bool(self.h.getBasis().valid),
                'simplex_iterations':int(info.simplex_iteration_count),'native_highs_version':self.native_version}
        if run_status!=highspy.HighsStatus.kOk or self.h.getModelStatus()!=highspy.HighsModelStatus.kOptimal or not solution.dual_valid:
            return {**common,'status':'uncertain','lower_bound':None,'certificate':None,
                    'reason':self.h.modelStatusToString(self.h.getModelStatus()),'seconds':time.perf_counter()-begin,'verify_seconds':0.}
        dual=np.asarray(solution.row_dual)
        # Positive row prices correspond to lower bounds in minimization.
        eta_lo=np.maximum(dual,0.);eta_hi=np.maximum(-dual,0.)
        verify_start=time.perf_counter()
        try:
            proof=self.checker.certificate(c.tolist(),eta_lo.tolist(),eta_hi.tolist(),threshold=float(threshold))
        except (ValueError,ArithmeticError) as failure:
            return {**common,'status':'uncertain','lower_bound':None,'certificate':None,'reason':str(failure),
                    'seconds':time.perf_counter()-begin,'verify_seconds':time.perf_counter()-verify_start}
        return {**common,'status':'certified' if proof['claimed_lower']>threshold else 'unresolved',
                'lower_bound':proof['claimed_lower'],'certificate':proof,
                'primal_diagnostic':float(c@np.asarray(solution.col_value)),
                'verify_seconds':time.perf_counter()-verify_start,'seconds':time.perf_counter()-begin}
