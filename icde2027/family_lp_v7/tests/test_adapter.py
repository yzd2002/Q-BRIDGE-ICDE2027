"""Synthetic-only bridge/adapter acceptance; no study input or old answer reads."""
import copy
import ctypes
from decimal import localcontext
from fractions import Fraction as Q
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'icde2027/family_lp_v7/adapter.py'
a=types.ModuleType('family_lp_adapter_synthetic_tests');a.__file__=str(SOURCE)
exec(compile(SOURCE.read_bytes(),str(SOURCE),'exec'),a.__dict__)
DEFAULT_LIBRARY=ROOT/'runs/icde2027/analysis/family_lp_v7_implementation/build_attempt_04/libfamily_lp_v7.so'
DEFAULT_SHA='3c89dda0b8a65db8b16b28a4a988153a4a4c73f01f881102b02a2d68f0742ee0'
GENERAL_SHA='dc7ac103796a946a4a481a03b5abad4167d9a8fbc559134ceff1de062697cf79'
LIBRARY=Path(os.environ.get('FAMILY_LP_TEST_LIBRARY',str(DEFAULT_LIBRARY)))
LIBRARY_SHA=os.environ.get('FAMILY_LP_TEST_LIBRARY_SHA256',DEFAULT_SHA)


def program(lo,hi,rows=()):return {'lower':list(lo),'upper':list(hi),'rows':list(rows)}
def row(c,lo,hi):return {'coefficients':list(c),'lower':lo,'upper':hi}
def raw_lp(c,lo,hi,rows=()):return {**program(lo,hi,rows),'objective':list(c)}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=a.ExactLP(LIBRARY,expected_library_sha256=LIBRARY_SHA,
            receiver_path=ROOT/'icde2027/partition_capacity_v6/verify.py',
            general_receiver_path=ROOT/'icde2027/family_lp_v7/verify_general.py',
            expected_general_receiver_sha256=GENERAL_SHA)

    def timed(self,result):
        t=result['timing_ns']
        keys=('model_ns','solve_ns','export_ns','receiver_ns','overhead_ns')
        self.assertTrue(all(type(t[k]) is int and t[k]>=0 for k in keys))
        self.assertEqual(sum(t[k] for k in keys),t['total_call_ns'])
        if result['native_status'] is not None:
            n=result['native_status']
            self.assertGreaterEqual(n['total_ns'],sum(n[k] for k in ('model_ns','solve_ns','export_ns')))

    def refuses(self,function,*args,**kwargs):
        with self.assertRaises(a.SolveError) as cm:function(*args,**kwargs)
        value=cm.exception.diagnostics
        self.assertFalse(value['complete']);self.assertEqual(value['status'],'refused')
        self.timed(value)
        return value

    def test_lower_side_dual_sign(self):
        r=self.s.solve_fixed_objective(program([0],[1],[row([1],Q(1,3),1)]),[1])
        self.assertEqual(r['certificate']['optimum'],'1/3')
        self.assertEqual(r['certificate']['original_row_lower_dual'],['1/1'])
        self.assertEqual(r['certificate']['original_row_upper_dual'],['0/1'])
        self.timed(r)

    def test_upper_side_dual_sign(self):
        r=self.s.solve_fixed_objective(program([0],[1],[row([1],0,Q(2,3))]),[-1])
        self.assertEqual(r['certificate']['optimum'],'-2/3')
        self.assertEqual(r['certificate']['original_row_lower_dual'],['0/1'])
        self.assertEqual(r['certificate']['original_row_upper_dual'],['1/1'])

    def test_signed_nondyadic_row(self):
        r=self.s.solve_fixed_objective(program([0],[1],[row([Q(-2,3)],Q(-2,3),Q(-2,9))]),[1])
        self.assertEqual(r['certificate']['point'],['1/3'])
        self.assertEqual(r['solution']['row_duals'],['-3/2'])

    def test_bound_reduced_costs(self):
        r=self.s.solve_fixed_objective(program([Q(1,3),Q(1,5)],[Q(2,3),Q(4,5)]),[3,-2])
        self.assertEqual(r['certificate']['point'],['1/3','4/5'])
        self.assertEqual(r['solution']['reduced_costs'],['3/1','-2/1'])
        self.assertEqual(r['certificate']['optimum'],'-3/5')

    def test_fixed_objective_scale_not_normalized(self):
        p=program([Q(1,3),0],[1,Q(1,4)])
        r=self.s.solve_fixed_objective(p,[Q(3,7),Q(-5,7)])
        scaled=self.s.solve_fixed_objective(p,[Q(9,7),Q(-15,7)])
        self.assertEqual(a.q(scaled['certificate']['optimum']),3*a.q(r['certificate']['optimum']))

    def test_zero_fixed_objective_still_feasible(self):
        p=program([0,0],[1,1],[row([1,1],Q(7,5),Q(7,5))])
        r=self.s.solve_fixed_objective(p,[0,0])
        self.assertEqual(r['certificate']['optimum'],'0/1')
        self.assertEqual(sum(map(a.q,r['certificate']['point'])),Q(7,5))

    def test_zero_equality_and_redundant_rows_retained(self):
        rows=[row([0,0],0,0),row([1,1],1,1),row([2,2],2,2),row([0,0],-1,1)]
        r=self.s.solve_fixed_objective(program([0,0],[1,1],rows),[1,-1])
        self.assertEqual(r['certificate']['point'],['0/1','1/1'])
        self.assertEqual(len(r['certificate']['original_row_lower_dual']),4)
        self.assertEqual(len(r['certificate']['original_row_upper_dual']),4)

    def test_known_partition_capacity(self):
        p=program([Q(3,4),0],[1,Q(1,4)])
        r=self.s.solve_family(p,[2],[1],[[Q(1,3),Q(2,3)]])
        self.assertEqual(r['certificate']['capacity'],'1/3')
        self.assertEqual(r['certificate']['policy'],[['1/1','0/1']])
        self.assertEqual(r['receiver_receipt']['scope'],'original-program-optimality-only')
        self.assertEqual(r['certificate']['contexts'],[{}])
        self.assertEqual(r['certificate']['internal_check'],{})

    def test_general_minimal_certificate(self):
        p=program([Q(3,4),0],[1,Q(1,4)])
        r=self.s.solve_general_family(p,[2],[1],[[Q(1,3),Q(2,3)]])
        self.assertEqual(r['certificate']['capacity'],'1/3')
        self.assertEqual(set(r['certificate']),{'schema_version','input_sha256','point','policy',
            'original_row_lower_dual','original_row_upper_dual','capacity'})
        self.assertTrue(r['receiver_receipt']['verified'])

    def test_zero_weight_has_reference_policy_and_feasible_point(self):
        p=program([Q(3,4),0,0,Q(2,3)],[1,Q(1,4),Q(1,3),1],
            [row([0,0,1,1],1,1)])
        ref=[[Q(1,3),Q(2,3)],[Q(2,5),Q(3,5)]]
        r=self.s.solve_general_family(p,[2,2],[1,0],ref)
        self.assertEqual(r['certificate']['policy'][1],['2/5','3/5'])
        self.assertEqual(r['epigraph_gamma'][2:],['0/1','0/1'])
        self.assertEqual(sum(map(a.q,r['certificate']['point'][2:])),1)

    def test_infeasible_zero_weight_block_is_refused(self):
        p=program([0,0],[1,1],[row([0,1],2,3)])
        r=self.refuses(self.s.solve_general_family,p,[1,1],[1,0],[[1],[1]])
        self.assertIsNotNone(r['native_status'])
        self.assertEqual(r['native_status']['outcome'],2)

    def test_ties_single_actions_zero_reference(self):
        p=program([Q(2,3),Q(2,3),Q(1,4)],[Q(2,3),Q(2,3),Q(1,4)])
        r=self.s.solve_general_family(p,[2,1],[Q(2,3),Q(1,3)],[[0,1],[1]])
        self.assertEqual(r['certificate']['capacity'],'0/1')
        self.assertEqual(r['certificate']['policy'][1],['1/1'])
        self.assertEqual(r['native_solution']['reduced_costs'][-2:],['0/1','0/1'])

    def test_free_t_stationarity_and_context_mass(self):
        p=program([Q(3,4),0,0,Q(2,3)],[1,Q(1,4),Q(1,3),1])
        r=self.s.solve_general_family(p,[2,2],[Q(2,7),Q(5,7)],[[Q(1,3),Q(2,3)],[Q(3,5),Q(2,5)]])
        gamma=list(map(a.q,r['epigraph_gamma']))
        self.assertEqual(sum(gamma[:2]),Q(2,7));self.assertEqual(sum(gamma[2:]),Q(5,7))
        self.assertEqual(r['native_solution']['reduced_costs'][-2:],['0/1','0/1'])

    def test_near_binary64_equal_coefficients_survive(self):
        eps=Q(1,2**60)
        self.assertEqual(float(1+eps),1.0)  # Test description only; never an LP input.
        p=program([0,0],[1,1],[row([1,1],1,1),row([1,1+eps],1+eps/3,1+eps/3)])
        r=self.s.solve_general_family(p,[2],[1],[[Q(1,2),Q(1,2)]])
        self.assertEqual(r['certificate']['point'],['2/3','1/3'])
        self.assertEqual(r['certificate']['capacity'],'1/6')
        self.assertTrue(r['native_solution']['model_roundtrip_exact'])

    def test_signed_overlap_cross_context(self):
        p=program([0,0,0,0],[1,1,1,1],[row([1,1,0,0],1,1),
            row([1,-1,0,0],Q(1,3),Q(1,3)),row([1,0,-1,0],0,0),
            row([0,1,0,-1],0,0)])
        r=self.s.solve_general_family(p,[2,2],[Q(1,4),Q(3,4)],[[Q(1,2),Q(1,2)],[1,0]])
        self.assertEqual(r['certificate']['point'],['2/3','1/3','2/3','1/3'])
        self.assertEqual(r['certificate']['capacity'],'1/24')

    def test_partition_receiver_domain_not_weakened(self):
        p=program([0,0],[1,1],[row([1,1],1,1),row([1,0],0,1)])
        r=self.refuses(self.s.solve_family,p,[2],[1],[[Q(1,2),Q(1,2)]])
        self.assertIsNone(r['native_status'])
        accepted=self.s.solve_general_family(p,[2],[1],[[Q(1,2),Q(1,2)]])
        self.assertTrue(accepted['receiver_receipt']['verified'])

    def test_original_objects_passed_to_independent_receiver(self):
        p=program([Q(3,4),0],[1,Q(1,4)]);sizes=[2];w=[1];ref=[[Q(1,3),Q(2,3)]]
        original=self.s.general_receiver.verify_family_certificate
        seen=[]
        def spy(*args,**kw):
            seen.append((args,kw));return original(*args,**kw)
        with patch.object(self.s.general_receiver,'verify_family_certificate',spy):
            r=self.s.solve_general_family(p,sizes,w,ref)
        for actual,expected in zip(seen[0][0][:4],(p,sizes,w,ref)):self.assertIs(actual,expected)
        self.assertEqual(seen[0][1]['expected_input_sha256'],r['input_sha256'])

    def test_request_mutation_before_receiver_fails(self):
        p=program([Q(3,4),0],[1,Q(1,4)])
        original=self.s._native
        def mutate(*args,**kw):
            result=original(*args,**kw);p['lower'][0]=Q(2,3);return result
        with patch.object(self.s,'_native',mutate):
            r=self.refuses(self.s.solve_general_family,p,[2],[1],[[Q(1,3),Q(2,3)]])
        self.assertEqual(r['phase'],'receiver')

    def test_changed_fixed_request_cannot_reuse_certificate(self):
        p=program([Q(1,3)],[1]);r=self.s.solve_fixed_objective(p,[1])
        with self.assertRaises(self.s.general_receiver.VerificationError):
            self.s.general_receiver.verify_fixed_certificate(p,[2],r['certificate'],
                expected_input_sha256=r['input_sha256'])

    def test_raw_lp_is_explicitly_not_independently_accepted(self):
        r=self.s.solve_linear_program(raw_lp([1],[0],[1],[row([1],Q(1,3),1)]))
        self.assertEqual(r['solution']['objective'],'1/3')
        self.assertIsNone(r['receiver_receipt'])
        self.assertEqual(r['status'],'exact_optimal_pending_independent_checker')
        self.assertNotIn('model_wire_roundtrip',r['solution'])
        self.assertEqual(r['solution']['model_roundtrip_sha256'],
            hashlib.sha256(a.native_wire(a.normalize_lp(raw_lp([1],[0],[1],[row([1],Q(1,3),1)])))).hexdigest())

    def test_unbounded_raw_refuses_optimal_status(self):
        r=self.refuses(self.s.solve_linear_program,raw_lp([-1],[None],[None]))
        self.assertEqual(r['native_status']['outcome'],2)
        self.assertGreater(r['timing_ns']['solve_ns'],0)
        self.assertGreater(r['timing_ns']['export_ns'],0)

    def test_infeasible_raw_refuses_optimal_status(self):
        self.refuses(self.s.solve_linear_program,raw_lp([1],[0],[1],[row([0],1,2)]))

    def test_float_bool_and_nonfinite_input_rejected(self):
        for value in (True,0.5,float('nan'),float('inf'),'1e-4','NaN'):
            with self.subTest(value=str(value)):
                r=self.refuses(self.s.solve_linear_program,raw_lp([value],[0],[1]))
                self.assertIsNone(r['native_status'])

    def test_malformed_dimensions_bounds_and_fields(self):
        cases=[raw_lp([],[],[]),raw_lp([1],[1],[0]),raw_lp([1],[0,0],[1]),
            raw_lp([1],[0],[1],[row([1,2],0,1)]),raw_lp([1],[0],[1],[row([1],2,1)]),
            {**raw_lp([1],[0],[1]),'unknown':1}]
        for value in cases:
            with self.subTest(value=value):self.refuses(self.s.solve_linear_program,value)

    def test_probability_inputs_not_silently_normalized(self):
        p=program([0,0],[1,1])
        for w,ref in (([2],[[Q(1,2),Q(1,2)]]),([1],[[1,1]]),([1],[[Q(-1,2),Q(3,2)]])):
            with self.subTest(w=w,ref=ref):
                r=self.refuses(self.s.solve_general_family,p,[2],w,ref)
                self.assertIsNone(r['native_status'])

    def test_general_request_canonical_strings_only(self):
        p=program(['0'],[1])
        r=self.refuses(self.s.solve_fixed_objective,p,[1]);self.assertIsNone(r['native_status'])
        r=self.s.solve_fixed_objective(program(['0/1'],['1/1']),['1/1'])
        self.assertEqual(r['certificate']['optimum'],'0/1')

    def test_general_box_contract_zero_one(self):
        self.refuses(self.s.solve_fixed_objective,program([-1],[1]),[1])
        self.refuses(self.s.solve_general_family,program([0],[2]),[1],[1],[[1]])

    def test_wrong_library_pin_has_setup_failure_timing(self):
        with self.assertRaises(a.SolveError) as cm:a.ExactLP(LIBRARY,expected_library_sha256='0'*64)
        d=cm.exception.diagnostics
        self.assertEqual(d['phase'],'setup');self.assertGreater(d['timing_ns']['setup_ns'],0)

    def test_wrong_general_receiver_pin_rejected(self):
        with self.assertRaises(a.SolveError):
            a.ExactLP(LIBRARY,expected_library_sha256=LIBRARY_SHA,
                general_receiver_path=ROOT/'icde2027/family_lp_v7/verify_general.py',expected_general_receiver_sha256='0'*64)

    def test_missing_receiver_never_claims_independent_acceptance(self):
        bare=a.ExactLP(LIBRARY,expected_library_sha256=LIBRARY_SHA)
        self.refuses(bare.solve_fixed_objective,program([0],[1]),[1])
        self.refuses(bare.solve_family,program([0],[1]),[1],[1],[[1]])
        self.refuses(bare.solve_general_family,program([0],[1]),[1],[1],[[1]])

    def test_timer_parameter_and_scope(self):
        for limit in (0,-1,True,0.5,float('nan'),object(),2**64):
            with self.subTest(limit=str(limit)):
                failure=self.refuses(self.s.solve_linear_program,raw_lp([1],[0],[1]),native_time_limit_ns=limit)
                json.dumps(failure,allow_nan=False)
        r=self.s.solve_linear_program(raw_lp([1],[0],[1]),native_time_limit_ns=10_000_000_000)
        self.assertIn('optimize-only',r['solver']['native_limit_scope'])
        self.assertIn('total_call_ns',r['timing_ns']);self.assertNotIn('total_service_ns',r['timing_ns'])
        self.timed(r)

    def test_finite_bound_infinity_collision_rejected(self):
        r=self.refuses(self.s.solve_linear_program,raw_lp([1],[0],[10**101]))
        self.assertEqual(r['native_status']['outcome'],1)

    def test_native_wire_invalid_tokens_and_sparse_structure(self):
        wires=[b'',b'WRONG\n1 0\n1/1 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n0 0\n',b'FAMILY_LP_EXACT/1\n2147483648 0\n',
            b'FAMILY_LP_EXACT/1\n1 0\n0.5 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n1 0\n1/0 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n1 0\n2/2 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n1 0\n+1/1 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n1 0\n-0/1 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n1 0\n01/1 0/1 1/1\n',
            b'FAMILY_LP_EXACT/1\n1 0\n1/1 0/1 1/1\ntrailing',
            b'FAMILY_LP_EXACT/1\n1 0\n1/1 0/1 1/1\x00',
            b'FAMILY_LP_EXACT/1\n2 1\n1/1 0/1 1/1\n1/1 0/1 1/1\n0/1 1/1 2 0 1/1 0 1/1',
            b'FAMILY_LP_EXACT/1\n1 1\n1/1 0/1 1/1\n0/1 1/1 1 0 0/1',
            b'FAMILY_LP_EXACT/1\n1 1\n1/1 0/1 1/1\n0/1 1/1 1 1 1/1']
        for wire in wires:
            with self.subTest(wire=wire):
                native=a._NativeResult()
                try:
                    code=self.s.lib.flp_solve_v1(wire,len(wire),0,ctypes.byref(native))
                    self.assertNotEqual(code,0);self.assertEqual(code,native.outcome)
                    self.assertGreater(native.total_ns,0)
                    self.assertIsNotNone(json.loads(ctypes.string_at(native.payload,native.payload_size)))
                finally:self.s.lib.flp_release_v1(ctypes.byref(native))
                self.assertFalse(native.payload)

    def test_solver_model_readback_crosses_abi_exactly(self):
        lp=raw_lp([Q(1,7),Q(-2,11)],[0,None],[1,None],[row([Q(1,3),Q(-2,5)],Q(1,9),Q(1,9))])
        wire=a.native_wire(a.normalize_lp(lp));native=a._NativeResult()
        try:
            code=self.s.lib.flp_solve_v1(wire,len(wire),0,ctypes.byref(native));self.assertEqual(code,0)
            result=json.loads(ctypes.string_at(native.payload,native.payload_size))
            self.assertEqual(result['model_wire_roundtrip'].encode('ascii'),wire)
        finally:self.s.lib.flp_release_v1(ctypes.byref(native))

    def test_chunk_parser_and_low_precision_exact_encoding(self):
        limit=sys.get_int_max_str_digits();big=10**5000+123456789
        with localcontext() as ctx:
            ctx.prec=2
            for value in (Q(big,7),Q(-big,11),Q(7,big),Q(0)):
                wire=a.render(value);self.assertEqual(a.q(wire),value)
        self.assertEqual(sys.get_int_max_str_digits(),limit)

    def test_random_fixed_point_family_cases(self):
        rng=random.Random(721)
        for case in range(30):
            sizes=[rng.randint(1,3),rng.randint(1,3)];n=sum(sizes)
            point=[Q(rng.randrange(8),7) for _ in range(n)]
            weight=Q(rng.randrange(6),5);weights=[weight,1-weight]
            reference=[]
            for size in sizes:
                counts=[rng.randrange(5) for _ in range(size)]
                if sum(counts)==0:counts[0]=1
                reference.append([Q(v,sum(counts)) for v in counts])
            rows=[]
            for _ in range(3):
                coeff=[Q(rng.randrange(-3,4),3) for _ in range(n)]
                value=sum((c*m for c,m in zip(coeff,point)),Q(0));rows.append(row(coeff,value,value))
            p=program(point,point,rows);expected=Q(0);offset=0
            for size,w,ref in zip(sizes,weights,reference):
                local=point[offset:offset+size];offset+=size
                expected+=w*(max(local)-sum((m*pi for m,pi in zip(local,ref)),Q(0)))
            with self.subTest(case=case):
                r=self.s.solve_general_family(p,sizes,weights,reference)
                self.assertEqual(a.q(r['certificate']['capacity']),expected)
                self.assertEqual(list(map(a.q,r['certificate']['point'])),point)

    def test_no_partition_constructor_import(self):
        self.assertNotIn('icde2027.partition_capacity_v6.core',sys.modules)
        self.assertNotIn('family_lp_v7_original_receiver.core',sys.modules)

    def test_repeated_calls_do_not_reuse_objective_or_answer(self):
        p=program([Q(1,3)],[Q(2,3)])
        first=self.s.solve_fixed_objective(p,[1])
        second=self.s.solve_fixed_objective(p,[-1])
        third=self.s.solve_fixed_objective(p,[1])
        self.assertEqual(first['certificate']['point'],['1/3'])
        self.assertEqual(second['certificate']['point'],['2/3'])
        self.assertEqual(first['certificate'],third['certificate'])

    def test_receiver_failure_is_timed_and_not_success(self):
        p=program([0],[1])
        def reject(*args,**kwargs):raise ValueError('synthetic receiver refusal')
        with patch.object(self.s.general_receiver,'verify_fixed_certificate',reject):
            failure=self.refuses(self.s.solve_fixed_objective,p,[1])
        self.assertEqual(failure['phase'],'receiver')
        self.assertGreater(failure['timing_ns']['receiver_ns'],0)

    def test_accepted_inputs_are_not_mutated(self):
        p=program([0,0],[1,1],[row([1,-1],Q(1,3),Q(1,3))])
        sizes=[2];w=[1];ref=[[Q(1,2),Q(1,2)]]
        before=copy.deepcopy((p,sizes,w,ref))
        self.s.solve_general_family(p,sizes,w,ref)
        self.assertEqual((p,sizes,w,ref),before)

    def test_native_model_readback_tamper_is_rejected(self):
        original=self.s.lib.flp_solve_v1
        def tamper(wire,length,limit,record):
            code=original(wire,length,limit,record)
            result=ctypes.cast(record,ctypes.POINTER(a._NativeResult)).contents
            text=ctypes.string_at(result.payload,result.payload_size)
            changed=text.replace(b'FAMILY_LP_EXACT/1',b'FAMILY_LP_EXACT/2')
            self.assertNotEqual(text,changed);self.assertEqual(len(text),len(changed))
            ctypes.memmove(result.payload,changed,len(changed))
            return code
        with patch.object(self.s.lib,'flp_solve_v1',tamper):
            failure=self.refuses(self.s.solve_linear_program,raw_lp([1],[0],[1]))
        self.assertIn('readback differs',failure['error'])

    def test_large_exact_input_passes_without_global_guard_change(self):
        guard=sys.get_int_max_str_digits();value=Q(1,10**5000+1)
        r=self.s.solve_fixed_objective(program([value],[value]),[1])
        self.assertEqual(a.q(r['certificate']['point'][0]),value)
        self.assertEqual(a.q(r['certificate']['optimum']),value)
        self.assertEqual(sys.get_int_max_str_digits(),guard)


if __name__=='__main__':unittest.main(verbosity=2)
