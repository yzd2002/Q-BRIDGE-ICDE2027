"""Materialized confidence views, component execution, and checked dual reuse.

All positive paths require a verified feasible witness for the immutable view.
The dual bank is mutable execution state and does not alter statistical scope.
"""
from __future__ import annotations
import hashlib
import json
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning

from .aggregate import _validate_program, randomized_intervals, solve_bound
from .proof_check import CheckedView, canonical_digest
from .source_intervals import observation_intervals, randomized_cell_projection, validate_counts


def _linprog(c, matrix, rhs, bounds, threads):
    options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10,
             "ipm_optimality_tolerance":1e-12}
    if threads is not None:
        options["threads"]=threads
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message=r"Unrecognized options detected: \{'threads': [0-9]+\}.*",category=OptimizeWarning)
        return linprog(c,A_ub=matrix,b_ub=rhs,bounds=bounds,method="highs",options=options)


def create_view(lower, upper, matrix, rhs_lower, rhs_upper, *, metadata=None, highs_threads=None):
    """Return a checked immutable-view document or an explicit uncertain status.

    Failure to find a verifiable point is not proof of infeasibility. This
    initial implementation may conservatively refuse a zero-volume feasible set.
    """
    begin=time.perf_counter()
    d=len(lower)
    _,lo,hi,t,a,b=_validate_program(np.zeros(d),lower,upper,matrix,rhs_lower,rhs_upper,
                                  allow_empty_intervals=True)
    base={"format":"qbridge-confidence-view-v1","lower":lo.tolist(),"upper":hi.tolist(),
          "rhs_lower":a.tolist(),"rhs_upper":b.tolist(),
          "matrix":{"indptr":t.indptr.tolist(),"indices":t.indices.tolist(),"data":t.data.tolist()},
          "metadata":metadata or {}}
    # A legal finite sample can make a truncated statistical interval empty.
    # Preserve those exact endpoints as a refusal; do not widen or swap them.
    if np.any(lo>hi) or np.any(a>b):
        return {"status":"uncertain","reason":"empty_statistical_intersection",
                "candidate_view":base,
                "empty_cell_intervals":np.flatnonzero(lo>hi).tolist(),
                "empty_group_intervals":np.flatnonzero(a>b).tolist(),
                "materialization_feasibility_seconds":time.perf_counter()-begin}
    try:
        proposal=solve_bound(np.zeros(d),lo,hi,t,a,b,export_proposal=True,verification="deferred",highs_threads=highs_threads)
    except (ValueError,ArithmeticError,RuntimeError):
        proposal=None
    def check(point):
        view={**base,"feasible_point":np.asarray(point,float).tolist()}
        try:
            checked=CheckedView(view)
            return {"status":"ready","view":view,"digest":checked.digest,
                    "materialization_feasibility_seconds":time.perf_counter()-begin}
        except ValueError:
            return None
    if proposal is not None and proposal.proposal is not None:
        accepted=check(proposal.proposal["primal"])
        if accepted:
            return accepted
    # Seek an interior point using a shared slack in mean units. The solver's
    # status never authorizes the view; exact checking of its point does.
    one=sparse.csr_matrix(np.ones((d,1)))
    rowone=sparse.csr_matrix(np.ones((t.shape[0],1)))
    eye=sparse.eye(d,format="csr")
    constraints=sparse.vstack([sparse.hstack([eye,one]),sparse.hstack([-eye,one]),
                               sparse.hstack([t,rowone]),sparse.hstack([-t,rowone])],format="csr")
    try:
        result=_linprog(np.r_[np.zeros(d),-1.],constraints,np.r_[hi,-lo,b,-a],
                        [(None,None)]*d+[(0,None)],highs_threads)
    except (ValueError,ArithmeticError,RuntimeError) as error:
        return {"status":"uncertain","reason":"feasibility_numerical_failure",
                "solver_report":str(error),"candidate_view":base,
                "materialization_feasibility_seconds":time.perf_counter()-begin}
    if result.success:
        accepted=check(result.x[:-1])
        if accepted:
            accepted["witness_search"]="slack_center"
            accepted["solver_slack"]=float(result.x[-1])
            return accepted
    return {"status":"uncertain","reason":"no_exact_feasible_witness",
            "solver_report":str(result.message),"candidate_view":base,
            "materialization_feasibility_seconds":time.perf_counter()-begin}


def materialize_counts(observational, randomized, *, rows, randomized_q, radius=2., alpha=.05,
                       transport_budget=0., consistency=True, randomized_method="group_score",
                       metadata=None, highs_threads=None):
    """Construct exactly one registered confidence set; no outcome-based choice.

    `group_score` and `cell_projection` are separately calibrated alternatives.
    This function does not choose their tighter interval after seeing the data.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    start=time.perf_counter()
    obs,rand=validate_counts(observational),validate_counts(randomized)
    if obs.shape!=rand.shape:
        raise ValueError("source count shapes differ")
    lower,upper,obs_info=observation_intervals(obs,radius,alpha/2,consistency=consistency)
    interval_obs=time.perf_counter()-start
    start=time.perf_counter()
    if randomized_method=="group_score":
        rlo,rhi,rand_info=randomized_intervals(rand,rows,randomized_q,alpha/2)
    elif randomized_method=="cell_projection":
        # The q/support declaration remains necessary for interpreting the R
        # source as randomized, even though this estimator conditions on A.
        randomized_intervals(np.zeros_like(rand),rows,randomized_q,alpha/2)
        rlo,rhi,rand_info=randomized_cell_projection(rand,rows,alpha/2)
    else:
        raise ValueError("unknown prespecified randomized interval construction")
    interval_random=time.perf_counter()-start
    rho=np.broadcast_to(np.asarray(transport_budget,float),rlo.shape)
    if np.any(rho<0) or not np.all(np.isfinite(rho)):
        raise ValueError("invalid transport tolerance")
    declaration={"alpha":alpha,"source_alpha":[alpha/2,alpha/2],"radius":radius,
                 "rho":rho.tolist(),"randomized_q":np.asarray(randomized_q).tolist(),
                 "observation_construction":obs_info,"randomized_construction":randomized_method,
                 "observational_counts_sha256":hashlib.sha256(obs.tobytes()).hexdigest(),
                 "randomized_counts_sha256":hashlib.sha256(rand.tobytes()).hexdigest(),
                 "counts_shape":list(obs.shape),"caller_declaration":metadata or {}}
    result=create_view(lower,upper,rows,rlo-rho,rhi+rho,metadata=declaration,highs_threads=highs_threads)
    result["timing"]={"observation_intervals":interval_obs,"randomized_intervals":interval_random,
                      "feasibility":result["materialization_feasibility_seconds"]}
    result["randomized_interval_diagnostics"]=rand_info
    return result


def connected_components(matrix):
    """Row--cell incidence components, including all unconstrained coordinates."""
    t=sparse.csr_matrix(matrix)
    d=t.shape[1]
    parent=np.arange(d)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=int(parent[i])
        return i
    touched=np.zeros(d,bool)
    for j in range(t.shape[0]):
        ids=t.indices[t.indptr[j]:t.indptr[j+1]]
        if not len(ids):
            continue
        touched[ids]=True
        root=find(int(ids[0]))
        for i in ids[1:]:
            other=find(int(i))
            if root!=other:
                parent[other]=root
    cell_groups={}
    for i in np.flatnonzero(touched):
        cell_groups.setdefault(find(int(i)),[]).append(int(i))
    row_groups={key:[] for key in cell_groups}
    zero_rows=[]
    for j in range(t.shape[0]):
        ids=t.indices[t.indptr[j]:t.indptr[j+1]]
        if len(ids):
            row_groups[find(int(ids[0]))].append(j)
        else:
            zero_rows.append(j)
    components=[(np.array(cells,dtype=int),np.array(row_groups[root],dtype=int)) for root,cells in cell_groups.items()]
    free=np.flatnonzero(~touched)
    if len(free):
        components.append((free,np.array([],dtype=int)))
    return components,zero_rows


class QueryEngine:
    """Evaluate many objectives against a single caller-trusted confidence view."""
    def __init__(self, trusted_view, *, plan="components", bank_size=16, reuse=True, highs_threads=None, compiled_reuse=False):
        start=time.perf_counter()
        if plan not in {"components","global","disjoint_or_global"} or type(bank_size) is not int or bank_size<0:
            raise ValueError("invalid execution plan / bank size")
        # Copy before checking: later mutation by the caller cannot change C.
        self._view=json.loads(json.dumps(trusted_view,allow_nan=False))
        self.checker=CheckedView(self._view)
        self.digest=self.checker.digest
        self.lower=np.asarray(self._view['lower']);self.upper=np.asarray(self._view['upper'])
        self.a=np.asarray(self._view['rhs_lower']);self.b=np.asarray(self._view['rhs_upper'])
        m=self._view['matrix']
        self.t=sparse.csr_matrix((m['data'],m['indices'],m['indptr']),shape=(len(self.a),len(self.lower)))
        self.plan=plan;self.bank_size=bank_size;self.reuse=reuse;self.highs_threads=highs_threads
        self.bank=[]
        self.compiled_reuse=compiled_reuse
        self.prepared_duals={}
        self.dual_preparation_seconds=0.
        self.components=[]
        if plan=='components':
            components,zero_rows=connected_components(self.t)
            # Exact feasibility has already checked any zero rows.
            simple_cells=[];simple_rows=[]
            for cells,rows in components:
                block=self.t[rows][:,cells].tocsr()
                if len(rows)<=1 and np.all(block.data>=0):
                    simple_cells.extend(cells.tolist());simple_rows.extend(rows.tolist())
                else:
                    self.components.append((cells,rows,block))
            # Batch all independent one-row/free blocks into a single sparse
            # knapsack proposal, avoiding one Python solver call per group.
            if simple_cells:
                cells=np.array(sorted(simple_cells),dtype=int);rows=np.array(sorted(simple_rows),dtype=int)
                self.components.append((cells,rows,self.t[rows][:,cells].tocsr()))
        self.initialization_seconds=time.perf_counter()-start

    def view_document(self):
        return json.loads(json.dumps(self._view))

    def _solve(self,c):
        d,j=len(c),len(self.a)
        if self.plan!='components':
            backend='highs' if self.plan=='global' else 'auto'
            proposal=solve_bound(c,self.lower,self.upper,self.t,self.a,self.b,backend=backend,
                                highs_threads=self.highs_threads,export_proposal=True,verification='deferred')
            if proposal.status!='proposal':
                return None,proposal.status
            return proposal.proposal,None
        eta_lo=np.zeros(j);eta_hi=np.zeros(j);point=np.zeros(d)
        for cells,rows,t in self.components:
            proposal=solve_bound(c[cells],self.lower[cells],self.upper[cells],t,self.a[rows],self.b[rows],
                                highs_threads=self.highs_threads,export_proposal=True,verification='deferred')
            if proposal.status!='proposal':
                return None,proposal.status
            point[cells]=proposal.proposal['primal']
            eta_lo[rows]=proposal.proposal['lower_multipliers']
            eta_hi[rows]=proposal.proposal['upper_multipliers']
        return {'primal':point.tolist(),'lower_multipliers':eta_lo.tolist(),'upper_multipliers':eta_hi.tolist()},None

    def query(self, objective, threshold=.001):
        start=time.perf_counter()
        c=np.asarray(objective,float)
        if c.shape!=self.lower.shape or not np.all(np.isfinite(c)) or not np.isfinite(threshold):
            raise ValueError("finite objective and threshold matching view required")
        norm=float(np.sum(np.abs(c)))
        if not np.isfinite(norm):
            raise ValueError("objective L1 norm exceeds finite implementation range")
        verify_time=0.;screen_time=0.
        if self.reuse:
            screen_start=time.perf_counter()
            # Zero dual is an explicit box shortcut, distinguished from reuse.
            candidates=[(np.zeros(len(self.a)),np.zeros(len(self.a)),'box_shortcut',None)]
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                for lo,hi in self.bank:
                    scaled_lo,scaled_hi=lo*norm,hi*norm
                    if np.all(np.isfinite(scaled_lo)) and np.all(np.isfinite(scaled_hi)):
                        candidates.append((scaled_lo,scaled_hi,'dual_reuse',(lo,hi)))
            values=[]
            for lo,hi,kind,base in candidates:
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    residual=c+self.t.T@(hi-lo)
                    val=float(self.a@lo-self.b@hi+np.minimum(residual*self.lower,residual*self.upper).sum())
                values.append(val if np.isfinite(val) else -np.inf)
            screen_time=time.perf_counter()-screen_start
            # Floating scores only choose work to check; they never certify.
            for index in np.argsort(values)[::-1]:
                if values[index] <= threshold:
                    continue
                lo,hi,kind,base=candidates[int(index)]
                verify_start=time.perf_counter()
                try:
                    if self.compiled_reuse and kind=='dual_reuse':
                        key=(base[0].tobytes(),base[1].tobytes())
                        if key not in self.prepared_duals or not self.checker.owns_prepared(
                                self.prepared_duals[key],base[0].tolist(),base[1].tolist()):
                            preparation_start=time.perf_counter()
                            prepared=self.checker.prepare_dual(base[0].tolist(),base[1].tolist())
                            self.dual_preparation_seconds+=time.perf_counter()-preparation_start
                            self.prepared_duals[key]=prepared
                            if len(self.prepared_duals)>max(1,self.bank_size):
                                self.prepared_duals.pop(next(iter(self.prepared_duals)))
                        certificate=self.checker.prepared_certificate(self.prepared_duals[key],c.tolist(),norm,
                            lower_multipliers=base[0].tolist(),upper_multipliers=base[1].tolist(),threshold=float(threshold))
                    else:
                        certificate=self.checker.certificate(c.tolist(),lo.tolist(),hi.tolist(),threshold=float(threshold))
                except (ValueError,ArithmeticError):
                    verify_time+=time.perf_counter()-verify_start
                    continue
                verify_time+=time.perf_counter()-verify_start
                if certificate['claimed_lower']>threshold:
                    return {'status':'certified','route':kind,'lower_bound':certificate['claimed_lower'],
                            'certificate':certificate,'primal_diagnostic':None,
                            'seconds':time.perf_counter()-start,'screen_seconds':screen_time,
                            'solve_seconds':0.,'verify_seconds':verify_time,'bank_entries':len(self.bank)}
        solve_start=time.perf_counter()
        try:
            proposal,error=self._solve(c)
        except (ValueError,ArithmeticError,RuntimeError) as failure:
            proposal,error=None,"solver_numerical_failure: "+str(failure)
        solve_time=time.perf_counter()-solve_start
        if proposal is None:
            return {'status':'uncertain','route':'solver','lower_bound':None,'certificate':None,
                    'reason':error,'seconds':time.perf_counter()-start,'screen_seconds':screen_time,
                    'solve_seconds':solve_time,'verify_seconds':verify_time,'bank_entries':len(self.bank)}
        verify_start=time.perf_counter()
        try:
            certificate=self.checker.certificate(c.tolist(),proposal['lower_multipliers'],proposal['upper_multipliers'],threshold=float(threshold))
        except (ValueError,ArithmeticError) as failure:
            verify_time+=time.perf_counter()-verify_start
            return {'status':'uncertain','route':'solver','lower_bound':None,'certificate':None,
                    'reason':'unverifiable_dual: '+str(failure),'seconds':time.perf_counter()-start,
                    'screen_seconds':screen_time,'solve_seconds':solve_time,'verify_seconds':verify_time,
                    'bank_entries':len(self.bank)}
        verify_time+=time.perf_counter()-verify_start
        if self.reuse and norm>0 and self.bank_size:
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                normalized_lo=np.asarray(proposal['lower_multipliers'])/norm
                normalized_hi=np.asarray(proposal['upper_multipliers'])/norm
            if np.all(np.isfinite(normalized_lo)) and np.all(np.isfinite(normalized_hi)):
                self.bank.append((normalized_lo,normalized_hi))
                if len(self.bank)>self.bank_size:
                    self.bank.pop(0)
        value=certificate['claimed_lower']
        return {'status':'certified' if value>threshold else 'unresolved','route':'solver',
                'lower_bound':value,'certificate':certificate,
                'primal_diagnostic':float(c@np.asarray(proposal['primal'])),
                'seconds':time.perf_counter()-start,'screen_seconds':screen_time,
                'solve_seconds':solve_time,'verify_seconds':verify_time,'bank_entries':len(self.bank)}
