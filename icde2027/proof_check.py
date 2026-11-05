"""Small optimizer-independent checker for a trusted, immutable view.

Standard library only. This verifies represented LP arithmetic and feasibility,
not the statistical quantile library or externally declared causal assumptions.
"""
from __future__ import annotations
from fractions import Fraction
from dataclasses import dataclass
import hashlib
import json
import math
import weakref


def canonical_digest(view):
    return hashlib.sha256(json.dumps(view,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _numbers(values):
    result=[]
    for value in values:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError("finite numeric coefficient required")
        result.append(Fraction.from_float(float(value)))
    return tuple(result)


@dataclass(frozen=True)
class PreparedDual:
    """Exact view-bound terms shared across objectives and positive scales.

    Exported proofs retain base multipliers and scale, so an external verifier
    can reconstruct the bound without trusting this prepared execution state.
    """
    view_digest: str
    lower: tuple
    upper: tuple
    offset: tuple
    constant: Fraction
    base_lower: tuple
    base_upper: tuple

    def exact_bound(self,objective,scale):
        c=_numbers(objective);factor=_numbers([scale])[0]
        if len(c)!=len(self.lower) or factor<0:
            raise ValueError('valid objective and nonnegative dual scale required')
        value=factor*self.constant
        for ci,ri,lo,hi in zip(c,self.offset,self.lower,self.upper):
            z=ci+factor*ri
            value+=z*(lo if z>=0 else hi)
        return value

    def certificate(self,objective,scale,*,threshold=None):
        value=math.nextafter(float(self.exact_bound(objective,scale)),-math.inf)
        if not math.isfinite(value):
            raise ValueError('finite lower bound required')
        if threshold is not None:
            _numbers([threshold])
        return {'format':'qbridge-scaled-dual-proof-v1','view_digest':self.view_digest,
                'objective':list(objective),'lower_multipliers':list(self.base_lower),
                'upper_multipliers':list(self.base_upper),'dual_scale':scale,
                'claimed_lower':value,'decision_threshold':threshold}


class CheckedView:
    """Precheck a caller-supplied view; keep private immutable rational tuples."""
    def __init__(self, view):
        if view.get("format") != "qbridge-confidence-view-v1":
            raise ValueError("unknown confidence view format")
        self.digest=canonical_digest(view)
        self.lower,self.upper=_numbers(view["lower"]),_numbers(view["upper"])
        self.rhs_lower,self.rhs_upper=_numbers(view["rhs_lower"]),_numbers(view["rhs_upper"])
        self.d,self.j=len(self.lower),len(self.rhs_lower)
        if not self.d or len(self.upper)!=self.d or len(self.rhs_upper)!=self.j:
            raise ValueError("invalid dimensions")
        if any(l<0 or u>1 or l>u for l,u in zip(self.lower,self.upper)):
            raise ValueError("invalid target-mean box")
        if any(l>u for l,u in zip(self.rhs_lower,self.rhs_upper)):
            raise ValueError("contradictory moment interval")
        ptr=view["matrix"]["indptr"]; ids=view["matrix"]["indices"]
        vals=_numbers(view["matrix"]["data"])
        if any(type(i) is not int for i in ptr+ids):
            raise ValueError("integer sparse indices required")
        if len(ptr)!=self.j+1 or ptr[0]!=0 or ptr[-1]!=len(ids) or len(ids)!=len(vals):
            raise ValueError("invalid sparse matrix lengths")
        if any(a>b for a,b in zip(ptr,ptr[1:])) or any(i<0 or i>=self.d for i in ids):
            raise ValueError("invalid sparse matrix indices")
        self.rows=tuple(tuple((ids[k],vals[k]) for k in range(ptr[j],ptr[j+1])) for j in range(self.j))
        point=_numbers(view["feasible_point"])
        if not self.is_feasible(point,already_rational=True):
            raise ValueError("whole-view feasibility witness does not verify")
        # Only states actually issued by this trusted checker may authorize a
        # cached positive path. Weak references do not retain evicted entries.
        self._issued_prepared=weakref.WeakValueDictionary()

    def is_feasible(self, point, *, already_rational=False):
        point=point if already_rational else _numbers(point)
        if len(point)!=self.d or any(not (l<=x<=u) for l,x,u in zip(self.lower,point,self.upper)):
            return False
        for row,lo,hi in zip(self.rows,self.rhs_lower,self.rhs_upper):
            value=sum((coef*point[i] for i,coef in row),Fraction(0))
            if not lo<=value<=hi:
                return False
        return True

    def exact_dual(self, objective, lower_multipliers, upper_multipliers):
        c=list(_numbers(objective));eta_lo=_numbers(lower_multipliers);eta_hi=_numbers(upper_multipliers)
        if len(c)!=self.d or len(eta_lo)!=self.j or len(eta_hi)!=self.j:
            raise ValueError("query/dual dimensions differ from trusted view")
        if any(v<0 for v in eta_lo+eta_hi):
            raise ValueError("dual multipliers must be nonnegative")
        value=sum((a*l-b*h for a,b,l,h in zip(self.rhs_lower,self.rhs_upper,eta_lo,eta_hi)),Fraction(0))
        for row,l,h in zip(self.rows,eta_lo,eta_hi):
            for i,coefficient in row:
                c[i]+=coefficient*(h-l)
        value+=sum((z*(l if z>=0 else u) for z,l,u in zip(c,self.lower,self.upper)),Fraction(0))
        return value

    def certificate(self, objective, lower_multipliers, upper_multipliers, *, threshold=None):
        exact=self.exact_dual(objective,lower_multipliers,upper_multipliers)
        value=math.nextafter(float(exact),-math.inf)
        if not math.isfinite(value):
            raise ValueError("finite lower bound required")
        if threshold is not None:
            _numbers([threshold])
        return {"format":"qbridge-dual-proof-v1","view_digest":self.digest,
                "objective":list(objective),"lower_multipliers":list(lower_multipliers),
                "upper_multipliers":list(upper_multipliers),"claimed_lower":value,
                "decision_threshold":threshold}

    def prepare_dual(self,lower_multipliers,upper_multipliers):
        eta_lo=_numbers(lower_multipliers);eta_hi=_numbers(upper_multipliers)
        if len(eta_lo)!=self.j or len(eta_hi)!=self.j or any(v<0 for v in eta_lo+eta_hi):
            raise ValueError('nonnegative base dual multipliers matching view required')
        residual=[Fraction(0) for _ in range(self.d)]
        constant=sum((a*l-b*h for a,b,l,h in zip(self.rhs_lower,self.rhs_upper,eta_lo,eta_hi)),Fraction(0))
        for row,l,h in zip(self.rows,eta_lo,eta_hi):
            for i,coefficient in row:
                residual[i]+=coefficient*(h-l)
        prepared=PreparedDual(self.digest,self.lower,self.upper,tuple(residual),constant,
                              tuple(float(v) for v in eta_lo),tuple(float(v) for v in eta_hi))
        self._issued_prepared[id(prepared)]=prepared
        return prepared

    def owns_prepared(self,prepared,lower_multipliers,upper_multipliers):
        return (isinstance(prepared,PreparedDual)
                and self._issued_prepared.get(id(prepared)) is prepared
                and prepared.view_digest==self.digest
                and prepared.lower is self.lower and prepared.upper is self.upper
                and prepared.base_lower==tuple(lower_multipliers)
                and prepared.base_upper==tuple(upper_multipliers))

    def prepared_certificate(self,prepared,objective,scale,*,lower_multipliers,upper_multipliers,threshold=None):
        if not self.owns_prepared(prepared,lower_multipliers,upper_multipliers):
            raise ValueError('prepared dual was not issued for this trusted view and base')
        return prepared.certificate(objective,scale,threshold=threshold)

    def verify(self, certificate, *, expected_objective=None, expected_threshold=None):
        if certificate.get("format") not in {"qbridge-dual-proof-v1","qbridge-scaled-dual-proof-v1"} or certificate.get("view_digest")!=self.digest:
            raise ValueError("proof is not bound to the trusted view")
        claim=_numbers([certificate["claimed_lower"]])[0]
        if expected_objective is not None and _numbers(expected_objective)!=_numbers(certificate['objective']):
            raise ValueError("proof objective differs from requested query")
        threshold=certificate.get('decision_threshold')
        if threshold is not None:
            threshold=_numbers([threshold])[0]
        if expected_threshold is not None and threshold!=_numbers([expected_threshold])[0]:
            raise ValueError("proof threshold differs from requested query")
        if certificate['format']=='qbridge-scaled-dual-proof-v1':
            prepared=self.prepare_dual(certificate['lower_multipliers'],certificate['upper_multipliers'])
            exact=prepared.exact_bound(certificate['objective'],certificate['dual_scale'])
        else:
            exact=self.exact_dual(certificate["objective"],certificate["lower_multipliers"],certificate["upper_multipliers"])
        if claim>exact:
            raise ValueError("claimed lower bound exceeds the checked weak dual")
        return {"valid_lp_lower_bound":True,"view_digest":self.digest,
                "claimed_lower":float(claim),"scientific_assumptions_verified":False,
                "certifies_declared_threshold":threshold is not None and claim>threshold,
                "requested_query_checked":expected_objective is not None and expected_threshold is not None}


def verify_certificate(trusted_view, certificate, *, expected_objective=None, expected_threshold=None):
    """Fresh external verification; neither input is executed as code."""
    return CheckedView(trusted_view).verify(certificate,expected_objective=expected_objective,
                                            expected_threshold=expected_threshold)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Check a policy lower-bound proof using only the Python standard library.')
    parser.add_argument('--view',type=Path,required=True,help='Caller-trusted confidence view JSON')
    parser.add_argument('--proof',type=Path,required=True,help='Ordinary or scaled dual proof JSON')
    parser.add_argument('--query',type=Path,required=True,help='Caller-requested JSON with objective and threshold fields')
    args=parser.parse_args()
    try:
        view=json.loads(args.view.read_text());proof=json.loads(args.proof.read_text());query=json.loads(args.query.read_text())
        result=verify_certificate(view,proof,expected_objective=query['objective'],expected_threshold=query['threshold'])
    except (ValueError,KeyError,TypeError,OSError,ArithmeticError) as error:
        parser.exit(2,f'Proof did not verify: {error}\n')
    print(json.dumps(result,indent=2,allow_nan=False))
