"""Thin exact LP adapter. It does not import the partition constructor.

The generic path has producer self-checks only. The supported family path also
uses the separately pinned, unchanged v6 receiver on the caller's original
request. No formal study, worker timeout, memory limiter or benchmarking harness
is implemented here.
"""
from contextlib import contextmanager
import ctypes
from decimal import Decimal
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import re
import time
import types

RECEIVER_SHA256='f9cd8a9b8f67a8794bd4c35b78064dc480bd5a94c1bdf47bf993f0f34f335fd6'
COMMIT='a00610534e044f35aa298059eb963312336c4c9c'


class InputError(ValueError):pass


class SolveError(RuntimeError):
    """Refusal/error with JSON-safe timings; no optimal result is returned."""
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        super().__init__(diagnostics['error'])


def need(condition,message):
    if not condition:raise InputError(message)


def q(value):
    if type(value) is int:return Q(value)
    if isinstance(value,Q):return Q(value.numerator,value.denominator)
    if type(value) is str and re.fullmatch(r'[+-]?[0-9]+(?:/[1-9][0-9]*)?',value):
        bits=value.split('/')
        # Chunked exact parsing leaves Python's process-wide decimal guard
        # unchanged. Accepted family/fixed requests are separately bounded by
        # their pinned independent receiver before this internal conversion.
        def integer(text):
            sign=-1 if text.startswith('-') else 1
            if text[:1] in ('+','-'):text=text[1:]
            result=0
            for offset in range(0,len(text),9):
                block=text[offset:offset+9];result=result*10**len(block)+int(block)
            return sign*result
        return Q(integer(bits[0]),integer(bits[1]) if len(bits)==2 else 1)
    raise InputError('Require int (not bool), Fraction, or integer/rational string; no float')


def render(value):
    # Exact conversion of already-computed integers; no input parser limit is
    # disabled and no binary64 conversion or Decimal arithmetic is performed.
    return format(Decimal(value.numerator),'f')+'/'+format(Decimal(value.denominator),'f')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()


def sequence(value,size=None):
    need(type(value) in (list,tuple),'Expected list or tuple')
    need(size is None or len(value)==size,'Dimension mismatch')
    return value


def normalize_lp(lp):
    need(type(lp) is dict and set(lp)=={'objective','lower','upper','rows'},'LP fields must be objective/lower/upper/rows')
    objective=tuple(q(v) for v in sequence(lp['objective']));n=len(objective)
    need(n>0,'Empty LP')
    lower=tuple(None if v is None else q(v) for v in sequence(lp['lower'],n))
    upper=tuple(None if v is None else q(v) for v in sequence(lp['upper'],n))
    need(all(lo is None or hi is None or lo<=hi for lo,hi in zip(lower,upper)),'Reversed box interval')
    rows=[]
    for raw in sequence(lp['rows']):
        need(type(raw) is dict and set(raw)=={'coefficients','lower','upper'},'Row fields differ')
        row=tuple(q(v) for v in sequence(raw['coefficients'],n))
        lo=None if raw['lower'] is None else q(raw['lower'])
        hi=None if raw['upper'] is None else q(raw['upper'])
        need(lo is None or hi is None or lo<=hi,'Reversed row interval')
        rows.append((row,lo,hi))
    return objective,lower,upper,tuple(rows)


def canonical_lp(parsed):
    objective,lower,upper,rows=parsed
    enc=lambda v:None if v is None else render(v)
    return {'schema_version':'family-lp-input/1','lp':{
        'objective':[render(v) for v in objective],'lower':[enc(v) for v in lower],'upper':[enc(v) for v in upper],
        'rows':[{'coefficients':[render(v) for v in a],'lower':enc(lo),'upper':enc(hi)} for a,lo,hi in rows]}}


def native_wire(parsed):
    objective,lower,upper,rows=parsed
    enc=lambda v:'*' if v is None else render(v)
    lines=['FAMILY_LP_EXACT/1',str(len(objective))+' '+str(len(rows))]
    lines.extend(render(c)+' '+enc(lo)+' '+enc(hi) for c,lo,hi in zip(objective,lower,upper))
    for a,lo,hi in rows:
        nz=[(i,v) for i,v in enumerate(a) if v]
        lines.append(' '.join([enc(lo),enc(hi),str(len(nz)),*(str(i)+' '+render(v) for i,v in nz)]))
    return ('\n'.join(lines)+'\n').encode('ascii')


class _NativeResult(ctypes.Structure):
    _fields_=[('abi_version',ctypes.c_uint32),('outcome',ctypes.c_int32),('solver_status',ctypes.c_int32),
        ('model_ns',ctypes.c_uint64),('solve_ns',ctypes.c_uint64),('export_ns',ctypes.c_uint64),
        ('total_ns',ctypes.c_uint64),('payload',ctypes.c_void_p),('payload_size',ctypes.c_size_t)]


class _Timing:
    def __init__(self):
        self.start=time.perf_counter_ns();self.phase='model'
        self.parts={name:0 for name in ('model_ns','solve_ns','export_ns','receiver_ns')}
        self.native=None;self.input_sha256=None
    @contextmanager
    def measure(self,name):
        self.phase=name;start=time.perf_counter_ns()
        try:yield
        finally:self.parts[name+'_ns']+=time.perf_counter_ns()-start
    def finish(self):
        elapsed=time.perf_counter_ns()-self.start
        return {**self.parts,'overhead_ns':max(0,elapsed-sum(self.parts.values())),
            'total_call_ns':elapsed,'native_total_ns':None if self.native is None else self.native['total_ns']}


class ExactLP:
    """Load a pinned shared library once; create one fresh LP per service call.

    Native time limits concern optimize() only. A future external service owner
    must enforce whole-call timeout, crash accounting, and memory limits.
    """
    def __init__(self,library_path,*,expected_library_sha256,receiver_path=None,
                 general_receiver_path=None,expected_general_receiver_sha256=None):
        started=time.perf_counter_ns()
        try:
            self._initialize(library_path,expected_library_sha256=expected_library_sha256,
                receiver_path=receiver_path,general_receiver_path=general_receiver_path,
                expected_general_receiver_sha256=expected_general_receiver_sha256)
        except Exception as error:
            elapsed=time.perf_counter_ns()-started
            raise SolveError({'schema_version':'family-lp-failure/1','status':'refused','complete':False,
                'mode':'initialization','phase':'setup','error':type(error).__name__+': '+str(error),
                'timing_ns':{'setup_ns':elapsed,'total_call_ns':elapsed}}) from error
        self.setup_ns=time.perf_counter_ns()-started

    def _initialize(self,library_path,*,expected_library_sha256,receiver_path=None,
                    general_receiver_path=None,expected_general_receiver_sha256=None):
        started=time.perf_counter_ns()
        self.library_path=Path(library_path).absolute()
        raw=self.library_path.read_bytes();self.library_sha256=hashlib.sha256(raw).hexdigest()
        need(type(expected_library_sha256) is str and self.library_sha256==expected_library_sha256,'Native library raw SHA mismatch')
        self.lib=ctypes.CDLL(str(self.library_path))
        need(hashlib.sha256(self.library_path.read_bytes()).hexdigest()==self.library_sha256,'Native library changed while loading')
        self.lib.flp_result_size_v1.restype=ctypes.c_size_t
        need(self.lib.flp_result_size_v1()==ctypes.sizeof(_NativeResult),'C ABI record size mismatch')
        self.lib.flp_build_info_v1.restype=ctypes.c_char_p
        self.build_info=self.lib.flp_build_info_v1().decode('ascii')
        need('commit='+COMMIT in self.build_info and 'exact-rational' in self.build_info,'Unreviewed native build identity')
        self.lib.flp_solve_v1.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_uint64,ctypes.POINTER(_NativeResult)]
        self.lib.flp_solve_v1.restype=ctypes.c_int
        self.lib.flp_release_v1.argtypes=[ctypes.POINTER(_NativeResult)]
        self.lib.flp_release_v1.restype=None
        self.receiver=None
        self.general_receiver=None
        if receiver_path is not None:
            path=Path(receiver_path).absolute();source=path.read_bytes()
            need(hashlib.sha256(source).hexdigest()==RECEIVER_SHA256,'Frozen common receiver SHA mismatch')
            module=types.ModuleType('family_lp_v7_original_receiver');module.__file__=str(path)
            exec(compile(source,str(path),'exec'),module.__dict__)
            self.receiver=module
        if general_receiver_path is not None:
            path=Path(general_receiver_path).absolute();source=path.read_bytes()
            need(type(expected_general_receiver_sha256) is str and
                 hashlib.sha256(source).hexdigest()==expected_general_receiver_sha256,'General independent receiver SHA mismatch')
            module=types.ModuleType('family_lp_v7_general_receiver');module.__file__=str(path)
            exec(compile(source,str(path),'exec'),module.__dict__)
            self.general_receiver=module
        self.setup_ns=time.perf_counter_ns()-started

    def _native(self,wire,limit,timer):
        need(limit is None or (type(limit) is int and 0<limit<2**64),'native_time_limit_ns must be a positive uint64 or None')
        result=_NativeResult();timer.phase='native'
        code=self.lib.flp_solve_v1(wire,len(wire),0 if limit is None else limit,ctypes.byref(result))
        timer.native={name:int(getattr(result,name)) for name in ('outcome','solver_status','model_ns','solve_ns','export_ns','total_ns')}
        for name in ('model_ns','solve_ns','export_ns'):timer.parts[name]+=timer.native[name]
        try:
            with timer.measure('export'):
                need(result.abi_version==1 and code==result.outcome,'Native status/ABI mismatch')
                need(result.payload and result.payload_size>0,'Native failure without payload; refuse')
                raw=ctypes.string_at(result.payload,result.payload_size)
                def pairs(items):
                    output={}
                    for key,value in items:
                        need(key not in output,'Duplicate native JSON key');output[key]=value
                    return output
                def bad(_value):raise InputError('Nonfinite native JSON number')
                value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
                if code!=0:raise RuntimeError(value.get('status','native_failure')+': '+value.get('error','unknown'))
                need(value.get('schema_version')=='family-lp-native-result/1' and value.get('status')=='exact_optimal'
                     and value.get('exact_original_primal_dual_equality') is True,'Exact native optimal certificate missing')
                need(value.get('model_wire_roundtrip')==wire.decode('ascii'),'Solver rational-model readback differs from the external model')
                # The exact imported coefficients have crossed the ABI and were
                # compared byte-for-byte. Keep the binding, not another full
                # copy of the input in the later service-result serialization.
                del value['model_wire_roundtrip']
                value['model_roundtrip_sha256']=hashlib.sha256(wire).hexdigest()
                value['model_roundtrip_exact']=True
                for name in ('primal','row_duals','reduced_costs'):
                    need(type(value[name]) is list,'Malformed native vector')
                    for token in value[name]:need(type(token) is str and render(q(token))==token,'Noncanonical native rational')
                need(type(value['objective']) is str and render(q(value['objective']))==value['objective'],'Native objective is not exact')
                return value
        finally:self.lib.flp_release_v1(ctypes.byref(result))

    def _attempt(self,mode,limit,operation):
        timer=_Timing()
        valid_limit=limit is None or (type(limit) is int and 0<limit<2**64)
        meta={'library_sha256':self.library_sha256,'build_info':self.build_info,
            'native_time_limit_ns':limit if valid_limit else None,
            'native_time_limit_parameter_valid':valid_limit,
            'native_limit_scope':'optimize-only; not whole service or memory',
            'one_time_setup_ns':self.setup_ns}
        try:
            need(valid_limit,'native_time_limit_ns must be a positive uint64 or None')
            result=operation(timer)
            result.update(schema_version='family-lp-service/1',mode=mode,timing_ns=timer.finish(),solver=meta,
                          native_status=timer.native)
            return result
        except Exception as error:
            diagnostic={'schema_version':'family-lp-failure/1','status':'refused','complete':False,'mode':mode,
                'phase':timer.phase,'error':type(error).__name__+': '+str(error),'input_sha256':timer.input_sha256,
                'timing_ns':timer.finish(),'native_status':timer.native,'solver':meta}
            raise SolveError(diagnostic) from error

    def solve_linear_program(self,lp,*,native_time_limit_ns=None):
        """Generic rational LP; returns exact quantities, not independent acceptance."""
        def operation(timer):
            with timer.measure('model'):
                parsed=normalize_lp(lp);canonical=canonical_lp(parsed);timer.input_sha256=digest(canonical)
                wire=native_wire(parsed)
            solution=self._native(wire,native_time_limit_ns,timer)
            with timer.measure('export'):
                need(len(solution['primal'])==len(solution['reduced_costs'])==len(parsed[0])
                     and len(solution['row_duals'])==len(parsed[3]),'Native dimension mismatch')
                return {'status':'exact_optimal_pending_independent_checker','input_sha256':timer.input_sha256,
                    'canonical_input':canonical,'solution':solution,'receiver_receipt':None,
                    'verification_scope':'native exact producer self-check only; raw LP is outside the family/fixed receiver API'}
        return self._attempt('general_lp',native_time_limit_ns,operation)

    def solve_fixed_objective(self,program,exact_objective,*,native_time_limit_ns=None):
        """min c*m on the entire original program, for later W2 decomposition.

        This does not reinterpret an old stored LP's primal-dual closure as slack.
        The later study must bind its normalized ideal objective separately.
        """
        def operation(timer):
            with timer.measure('model'):
                need(self.general_receiver is not None,'Pinned independent fixed-objective receiver is required')
                expected=self.general_receiver.fixed_input_digest(program,exact_objective)
                timer.input_sha256=expected
                need(type(program) is dict and set(program)=={'lower','upper','rows'},'Original program fields differ')
                parsed=normalize_lp({**program,'objective':exact_objective})
                need(all(v is not None for v in parsed[1]+parsed[2]),'Fixed original program requires finite boxes')
                need(all(lo is not None and hi is not None for _a,lo,hi in parsed[3]),'Original moment rows require both finite sides')
                canonical=canonical_lp(parsed);wire=native_wire(parsed)
            solution=self._native(wire,native_time_limit_ns,timer)
            with timer.measure('export'):
                signed=[q(v) for v in solution['row_duals']]
                certificate={'schema_version':'fixed-policy-optimum-certificate/1','input_sha256':expected,
                    'point':solution['primal'],'optimum':solution['objective'],
                    'original_row_lower_dual':[render(max(v,Q(0))) for v in signed],
                    'original_row_upper_dual':[render(max(-v,Q(0))) for v in signed]}
            with timer.measure('receiver'):
                receipt=self.general_receiver.verify_fixed_certificate(program,exact_objective,certificate,
                    expected_input_sha256=expected)
            return {'status':'verified_optimal','input_sha256':expected,'canonical_model':canonical,
                'solution':solution,'certificate':certificate,'receiver_receipt':receipt}
        return self._attempt('fixed_objective',native_time_limit_ns,operation)

    def solve_family(self,program,context_sizes,exact_weights,exact_reference,*,native_time_limit_ns=None):
        """Supported-partition W1 epigraph, accepted only by the frozen common receiver."""
        return self._solve_family(program,context_sizes,exact_weights,exact_reference,native_time_limit_ns,'partition')

    def solve_general_family(self,program,context_sizes,exact_weights,exact_reference,*,native_time_limit_ns=None):
        """General moment views, accepted only by the separate general receiver."""
        return self._solve_family(program,context_sizes,exact_weights,exact_reference,native_time_limit_ns,'general')

    def _solve_family(self,program,context_sizes,exact_weights,exact_reference,limit,kind):
        def operation(timer):
            with timer.measure('model'):
                receiver=self.receiver if kind=='partition' else self.general_receiver
                need(receiver is not None,'Pinned independent receiver is required for family acceptance')
                digest_function=receiver.input_digest if kind=='partition' else receiver.family_input_digest
                expected=digest_function(program,context_sizes,exact_weights,exact_reference)
                timer.input_sha256=expected
                sizes=tuple(context_sizes);weights=tuple(q(v) for v in exact_weights)
                references=tuple(tuple(q(v) for v in row) for row in exact_reference)
                n=sum(sizes);base=normalize_lp({**program,'objective':[0]*n})
                _unused,lower,upper,rows=base;number_of_contexts=len(sizes)
                objective=[-w*p for w,ref in zip(weights,references) for p in ref]+list(weights)
                lifted_rows=[(tuple(a)+(Q(0),)*number_of_contexts,lo,hi) for a,lo,hi in rows]
                offset=0
                for x,size in enumerate(sizes):
                    for i in range(offset,offset+size):
                        a=[Q(0)]*(n+number_of_contexts);a[i]=Q(1);a[n+x]=Q(-1)
                        lifted_rows.append((tuple(a),None,Q(0)))
                    offset+=size
                parsed=(tuple(objective),lower+(None,)*number_of_contexts,upper+(None,)*number_of_contexts,tuple(lifted_rows))
                wire=native_wire(parsed)
                canonical={'schema_version':'partition-capacity-input/1' if kind=='partition' else 'family-capacity-input/1',
                    'program':{'lower':[render(v) for v in lower],'upper':[render(v) for v in upper],
                        'rows':[{'coefficients':[render(v) for v in a],'lower':render(lo),'upper':render(hi)} for a,lo,hi in rows]},
                    'context_sizes':list(sizes),'exact_weights':[render(v) for v in weights],
                    'exact_reference':[[render(v) for v in ref] for ref in references]}
                if kind=='partition':need(digest(canonical)==expected,'Producer canonical request differs from common receiver digest')
            solution=self._native(wire,limit,timer)
            with timer.measure('export'):
                point=tuple(q(v) for v in solution['primal'][:n]);native_dual=tuple(q(v) for v in solution['row_duals'])
                need(len(solution['primal'])==len(solution['reduced_costs'])==n+number_of_contexts
                     and len(native_dual)==len(rows)+n,'Native epigraph dimension mismatch')
                signed=native_dual[:len(rows)];gamma=tuple(-v for v in native_dual[len(rows):])
                need(all(v>=0 for v in gamma),'Epigraph upper-row dual has wrong sign')
                need(all(q(v)==0 for v in solution['reduced_costs'][n:]),'Free-t stationarity failed')
                policy=[];c=[];common=Q(0);offset=0
                for x,size in enumerate(sizes):
                    local_gamma=gamma[offset:offset+size];local=point[offset:offset+size]
                    need(sum(local_gamma,Q(0))==weights[x],'Epigraph dual mass does not equal context weight')
                    if weights[x]==0:
                        need(all(v==0 for v in local_gamma),'Zero-weight gamma is not zero');policy.append(references[x])
                    else:policy.append(tuple(v/weights[x] for v in local_gamma))
                    c.extend(weights[x]*(qi-pi) for qi,pi in zip(policy[-1],references[x]))
                    common+=weights[x]*(max(local)-sum((p*m for p,m in zip(references[x],local)),Q(0)))
                    offset+=size
                eta=tuple(max(v,Q(0)) for v in signed);lam=tuple(max(-v,Q(0)) for v in signed)
                z=list(c);values=[];dual_lower=Q(0)
                for a,lo,hi,e,l in (r+(e,l) for r,e,l in zip(rows,eta,lam)):
                    values.append(sum((ai*mi for ai,mi in zip(a,point)),Q(0)))
                    dual_lower+=lo*e-hi*l
                    z=[zi+ai*(l-e) for zi,ai in zip(z,a)]
                dual_lower+=sum((min(zi*lo,zi*hi) for zi,lo,hi in zip(z,lower,upper)),Q(0))
                attained=sum((ci*mi for ci,mi in zip(c,point)),Q(0))
                need(dual_lower==attained==common==q(solution['objective']),'Family exact LB/attainment/F/native objective disagree')
                need(tuple(z)==tuple(q(v) for v in solution['reduced_costs'][:n]),'Original box reduced costs disagree')
                certificate={'schema_version':'partition-capacity-v6/1','input_sha256':expected,'canonical_input':canonical,
                    'point':[render(v) for v in point],'policy':[[render(v) for v in row] for row in policy],
                    'original_row_lower_dual':[render(v) for v in eta],'original_row_upper_dual':[render(v) for v in lam],
                    'original_row_values':[render(v) for v in values],'objective_coefficients':[render(v) for v in c],
                    'box_reduced_costs':[render(v) for v in z],'capacity':render(common),
                    'common_point_upper_bound':render(common),'policy_at_point':render(attained),
                    'policy_dual_lower_bound':render(dual_lower),'contexts':[{} for _ in sizes],'internal_check':{}}
                if kind=='general':
                    certificate={key:certificate[key] for key in ('input_sha256','point','policy',
                        'original_row_lower_dual','original_row_upper_dual','capacity')}
                    certificate['schema_version']='family-capacity-certificate/1'
            with timer.measure('receiver'):
                verify=receiver.verify_certificate if kind=='partition' else receiver.verify_family_certificate
                receipt=verify(program,context_sizes,exact_weights,exact_reference,
                    certificate,expected_input_sha256=expected)
            return {'status':'verified_optimal','input_sha256':expected,'certificate':certificate,
                'receiver_receipt':receipt,'native_solution':solution,
                'epigraph_gamma':[render(v) for v in gamma]}
        return self._attempt('family_epigraph' if kind=='partition' else 'general_family_epigraph',limit,operation)


__all__=['ExactLP','InputError','SolveError','RECEIVER_SHA256']
