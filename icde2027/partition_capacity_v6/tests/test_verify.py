"""Independently authored receiver and producer-review tests; synthetic only."""

import ast
import copy
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

from icde2027.partition_capacity_v6.core import solve, UnsupportedStructureError


ROOT = Path(__file__).resolve().parents[3]
VERIFY_PATH = ROOT / "icde2027/partition_capacity_v6/verify.py"
MATH_REVIEW = ROOT / "runs/icde2027/analysis/partition_capacity_math_review_v6"
COUNTS = {"exact_oracle_cases": 0, "exact_oracle_systems": 0, "exact_oracle_vertices": 0,
          "multicontext_cases": 0, "cross_request_rejections": 0,
          "request_rejections": 0, "certificate_rejections": 0,
          "unsupported_structure_checks": 0, "standalone_process_checks": 0}
ORACLE_RECORDS = []
MULTICONTEXT_RECORDS = []


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receiver = load_file_module(VERIFY_PATH, "independent_partition_receiver_tests")


def raw_program(lo, hi, rows=()):
    return {"lower":lo, "upper":hi, "rows":[
        {"coefficients":t, "lower":a, "upper":b} for t,a,b in rows]}


def lower_row_request():
    return (raw_program([0,0], ["1/3",1], [([1,1],1,1)]),
            [2], [1], [["1/2","1/2"]])


def upper_row_request():
    return (raw_program([0,0,1], [1,1,1], [([1,1,0],0,1)]),
            [3], [1], [["3/5","2/5",0]])


def two_context_request():
    return (raw_program([0]*4, ["1/3",1,"1/4",1],
                        [([1,1,0,0],1,1),([0,0,1,1],1,1)]),
            [2,2], ["1/3","2/3"], [["1/2","1/2"],["1/2","1/2"]])


def merge_requests(requests, weights):
    n=sum(len(r[0]['lower']) for r in requests)
    lo,hi,rows,sizes,refs=[],[],[],[],[]
    start=0
    for p,ss,ww,rr in requests:
        width=len(p['lower'])
        lo.extend(p['lower']);hi.extend(p['upper']);sizes.extend(ss);refs.extend(rr)
        for row in p['rows']:
            rows.append(([0]*start+list(row['coefficients'])+[0]*(n-start-width),
                         row['lower'],row['upper']))
        start+=width
    rows.insert(0,([0]*n,0,0))
    return raw_program(lo,hi,rows),sizes,weights,refs


class ReceiverTests(unittest.TestCase):
    def reject_certificate(self, request, certificate):
        with self.assertRaises(receiver.CertificateValidationError):
            receiver.verify_certificate(*request,certificate)
        COUNTS['certificate_rejections']+=1

    def reject_request(self, request, certificate):
        with self.assertRaises(receiver.RequestValidationError):
            receiver.verify_certificate(*request,certificate)
        COUNTS['request_rejections']+=1

    def test_known_exact_lower_and_upper_row_certificates(self):
        for request,value in [(lower_row_request(),F(1,6)),(upper_row_request(),F(2,5))]:
            cert=solve(*request)
            result=receiver.verify_certificate(*request,cert,
                        expected_input_sha256=receiver.input_digest(*request))
            self.assertEqual(F(result['capacity']),value)
            self.assertTrue(result['verified'])
            self.assertEqual(result['schema_version'],'partition-capacity-verification/1')
            self.assertEqual(result['unchecked_fields'],['contexts','internal_check'])
            self.assertEqual(result['scope'],'original-program-optimality-only')

    def test_digest_from_manual_canonical_request(self):
        request=lower_row_request()
        canonical={'schema_version':'partition-capacity-input/1',
                   'program':{'lower':['0/1','0/1'],'upper':['1/3','1/1'],
                              'rows':[{'coefficients':['1/1','1/1'],'lower':'1/1','upper':'1/1'}]},
                   'context_sizes':[2],'exact_weights':['1/1'],
                   'exact_reference':[['1/2','1/2']]}
        expected=hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=True,
                              separators=(',',':'),allow_nan=False).encode('ascii')).hexdigest()
        self.assertEqual(receiver.input_digest(*request),expected)
        equivalent=copy.deepcopy(request)
        equivalent[0]['upper'][0]='+02/6'
        equivalent[3][0]=['2/4','3/6']
        self.assertEqual(receiver.input_digest(*equivalent),expected)
        receiver.verify_certificate(*equivalent,solve(*request))

    def test_expected_digest_cannot_replace_trusted_request_binding(self):
        request=lower_row_request();cert=solve(*request)
        for digest in ('0'*64,'F'*64,True,'abc',123):
            with self.subTest(digest=digest):
                with self.assertRaises(receiver.RequestValidationError):
                    receiver.verify_certificate(*request,cert,expected_input_sha256=digest)
                COUNTS['request_rejections']+=1

    def test_cross_request_replacement_all_input_components(self):
        original=two_context_request()
        replacements=[]
        c=copy.deepcopy(original);c[0]['lower'][0]='1/6';replacements.append(c)
        c=copy.deepcopy(original);c[0]['upper'][0]='2/5';replacements.append(c)
        c=copy.deepcopy(original);c[0]['rows'][0]['coefficients'][0]=2;replacements.append(c)
        c=copy.deepcopy(original);c[0]['rows'][0]['lower']=0;replacements.append(c)
        c=copy.deepcopy(original);c[0]['rows'][0]['upper']=2;replacements.append(c)
        c=list(copy.deepcopy(original));c[1]=[4];c[2]=[1];c[3]=[['1/4']*4];replacements.append(c)
        c=copy.deepcopy(original);c[2][:]=['1/2','1/2'];replacements.append(c)
        c=copy.deepcopy(original);c[3][0][:]=[1,0];replacements.append(c)
        c=copy.deepcopy(original);c[0]['rows'].insert(0,{'coefficients':[0]*4,'lower':0,'upper':0});replacements.append(c)
        c=copy.deepcopy(original);c[0]['rows'].reverse();replacements.append(c)
        for changed in replacements:
            cert=solve(*changed)
            self.assertNotEqual(cert['input_sha256'],receiver.input_digest(*original))
            self.reject_certificate(original,cert)
            COUNTS['cross_request_rejections']+=1

    def test_embedded_input_and_hash_replaced_with_original_do_not_hide_wrong_proof(self):
        original=lower_row_request();other=copy.deepcopy(original)
        other[3][0][:]=[1,0]
        cert=solve(*other);correct=solve(*original)
        cert['canonical_input']=correct['canonical_input'];cert['input_sha256']=correct['input_sha256']
        self.reject_certificate(original,cert)
        COUNTS['cross_request_rejections']+=1

    def test_original_zero_row_deletion_is_detected(self):
        original=copy.deepcopy(lower_row_request())
        original[0]['rows'].append({'coefficients':[0,0],'lower':0,'upper':0})
        self.reject_certificate(original,solve(*lower_row_request()))
        COUNTS['cross_request_rejections']+=1

    def test_all_mathematical_top_level_fields_are_recomputed(self):
        request=lower_row_request()
        mutations=[
            lambda c:c.__setitem__('input_sha256','0'*64),
            lambda c:c['point'].__setitem__(0,'0/1'),
            lambda c:c['policy'][0].__setitem__(0,'1/1'),
            lambda c:c['original_row_lower_dual'].__setitem__(0,'0/1'),
            lambda c:c['original_row_lower_dual'].__setitem__(0,'-1/2'),
            lambda c:c['original_row_upper_dual'].__setitem__(0,'1/1'),
            lambda c:c['original_row_values'].__setitem__(0,'0/1'),
            lambda c:c['objective_coefficients'].__setitem__(0,'0/1'),
            lambda c:c['box_reduced_costs'].__setitem__(0,'0/1'),
        ]
        for field in ('capacity','common_point_upper_bound','policy_at_point','policy_dual_lower_bound'):
            mutations.append(lambda c,field=field:c.__setitem__(field,'1/7'))
        for mutate in mutations:
            cert=solve(*request);mutate(cert);self.reject_certificate(request,cert)

    def test_schema_keys_shapes_and_boolean_canonical_impostor(self):
        request=lower_row_request()
        mutations=[
            lambda c:c.__setitem__('schema_version','partition-capacity-v6/2'),
            lambda c:c.__setitem__('unbound_extra','anything'),
            lambda c:c.pop('capacity'),
            lambda c:c.__setitem__('point',['1/3']),
            lambda c:c.__setitem__('point',tuple(c['point'])),
            lambda c:c.__setitem__('policy',[]),
            lambda c:c['policy'].__setitem__(0,['1/1']),
            lambda c:c.__setitem__('original_row_lower_dual',[]),
            lambda c:c.__setitem__('original_row_upper_dual',['0/1','0/1']),
            lambda c:c.__setitem__('contexts',[]),
            lambda c:c.__setitem__('internal_check',[]),
            lambda c:c['canonical_input']['program']['upper'].__setitem__(0,'2/6'),
        ]
        for mutate in mutations:
            cert=solve(*request);mutate(cert);self.reject_certificate(request,cert)
        one=(raw_program([0],[1]),[1],[1],[[1]])
        cert=solve(*one);cert['canonical_input']['context_sizes'][0]=True
        self.reject_certificate(one,cert)

    def test_certificate_rationals_are_strict_canonical_strings(self):
        request=lower_row_request()
        for bad in (0,F(1,3),0.5,True,Decimal('0.3'),'1','+1/3','2/6','01/3',
                    '-0/1','1/03','0.3','1/0','1/-3',' 1/3','1/3 '):
            cert=solve(*request);cert['point'][0]=bad
            self.reject_certificate(request,cert)

    def test_trusted_input_has_no_float_bool_or_implicit_normalization(self):
        cert=solve(*lower_row_request())
        for bad in (0.5,True,Decimal('0.5'),'0.5','1e-2',' 1/2','1/2 ','1/0',
                    '1/-2','1/02','nan',None,object()):
            request=copy.deepcopy(lower_row_request());request[0]['lower'][0]=bad
            self.reject_request(request,cert)
        for p in ([1,1],[0,0],[-1,2],['1/3','1/3']):
            request=copy.deepcopy(lower_row_request());request[3][0][:]=p
            self.reject_request(request,cert)
        for w in ([2],[0],[-1],['999/1000']):
            request=copy.deepcopy(lower_row_request());request[2][:]=w
            self.reject_request(request,cert)

    def test_float_rejected_in_each_trusted_numeric_part(self):
        mutators=[lambda r:r[0]['upper'].__setitem__(0,0.5),
                  lambda r:r[0]['rows'][0]['coefficients'].__setitem__(0,1.0),
                  lambda r:r[0]['rows'][0].__setitem__('lower',1.0),
                  lambda r:r[0]['rows'][0].__setitem__('upper',1.0),
                  lambda r:r[2].__setitem__(0,1.0),
                  lambda r:r[3][0].__setitem__(0,0.5)]
        for mutate in mutators:
            request=copy.deepcopy(lower_row_request());mutate(request)
            self.reject_request(request,solve(*lower_row_request()))

    def test_malformed_request_dimensions_metadata_and_bounds(self):
        mutators=[lambda r:r.__setitem__(1,[]),lambda r:r.__setitem__(1,[0,2]),
                  lambda r:r.__setitem__(1,[True,1]),lambda r:r.__setitem__(1,[3]),
                  lambda r:r.__setitem__(1,'2'),lambda r:r.__setitem__(2,[]),
                  lambda r:r.__setitem__(3,[]),lambda r:r.__setitem__(3,[[1]]),
                  lambda r:r[0].__setitem__('upper',[1]),
                  lambda r:r[0].__setitem__('rows',None),
                  lambda r:r[0].__setitem__('metadata','unbound'),
                  lambda r:r[0]['rows'][0].__setitem__('metadata','unbound'),
                  lambda r:r[0]['rows'][0].__setitem__('coefficients',[1]),
                  lambda r:r[0]['rows'][0].__setitem__('lower',2),
                  lambda r:r[0]['lower'].__setitem__(0,-1),
                  lambda r:r[0]['upper'].__setitem__(0,2),
                  lambda r:r[0]['lower'].__setitem__(0,1)]
        for mutate in mutators:
            request=list(copy.deepcopy(lower_row_request()));mutate(request)
            self.reject_request(request,solve(*lower_row_request()))

    def test_unsupported_domain_examples_rejected_by_both_implementations(self):
        requests=[(raw_program([0,0,0],[1,1,1],[([1,0,1],1,1),([0,1,2],1,1)]),[3],[1],[[0,1,0]]),
                  (raw_program([0,0],[1,1],[([2,-1],1,1)]),[2],[1],[[0,1]]),
                  (raw_program([0]*4,[1,0,1,0],[([1,0,1,0],1,1)]),[2,2],['1/2']*2,[[0,1],[0,1]])]
        for request in requests:
            with self.assertRaises(UnsupportedStructureError):solve(*request)
            self.reject_request(request,solve(*lower_row_request()))
            COUNTS['unsupported_structure_checks']+=1

    def test_impossible_original_constraints_are_not_blessed(self):
        request=lower_row_request();cert=solve(*request)
        # Change both trusted input and its binding, leaving the proof impossible.
        changed=copy.deepcopy(request);changed[0]['rows'][0]['lower']=3;changed[0]['rows'][0]['upper']=4
        cert['canonical_input']['program']['rows'][0]['lower']='3/1'
        cert['canonical_input']['program']['rows'][0]['upper']='4/1'
        cert['input_sha256']=receiver.input_digest(*changed)
        self.reject_certificate(changed,cert)
        changed=copy.deepcopy(request);changed[0]['rows']=[{'coefficients':[0,0],'lower':1,'upper':1}]
        self.reject_request(changed,solve(*request))

    def test_zero_weight_context_still_requires_legal_policy(self):
        request=list(two_context_request());request[2]=[0,1]
        cert=solve(*request);cert['policy'][0]=['0/1','0/1']
        self.reject_certificate(request,cert)

    def test_original_duals_must_be_weighted_even_when_local_metadata_is_present(self):
        request=two_context_request();cert=solve(*request)
        cert['original_row_lower_dual'][0]='1/2'
        self.reject_certificate(request,cert)

    def test_zero_row_duals_need_nonnegativity_but_not_constructor_identity(self):
        request=copy.deepcopy(lower_row_request())
        request[0]['rows'].insert(0,{'coefficients':[0,0],'lower':0,'upper':0})
        cert=solve(*request)
        cert['original_row_lower_dual'][0]='4/3';cert['original_row_upper_dual'][0]='7/2'
        self.assertTrue(receiver.verify_certificate(*request,cert)['verified'])
        cert['original_row_lower_dual'][0]='-4/3'
        self.reject_certificate(request,cert)

    def test_auxiliary_claims_are_explicitly_unchecked(self):
        request=lower_row_request();cert=solve(*request)
        cert['contexts'][0]={'cap':'999/1','root':'-999/1','deficit':'999/1','unverified':True}
        cert['internal_check']={'independent_review':True,'kind':'a false auxiliary claim'}
        result=receiver.verify_certificate(*request,cert)
        self.assertTrue(result['verified'])
        self.assertEqual(set(result['unchecked_fields']),{'contexts','internal_check'})
        self.assertNotIn('contexts',result['checked_fields'])
        self.assertIn('no mathematical validation',result['unchecked_fields_notice'])
        cert['contexts'][0]['cap']=1.5
        self.reject_certificate(request,cert)  # JSON wire typing still applies.

    def test_valid_certificate_need_not_use_minimal_cap(self):
        request=(raw_program([0,0],[1,1]),[2],[1],[['1/2','1/2']])
        cert=solve(*request);cert['point']=['3/4','3/4']
        self.assertEqual(cert['contexts'][0]['cap'],'0/1')  # Deliberately stale auxiliary cap.
        self.assertEqual(receiver.verify_certificate(*request,cert)['capacity'],'0/1')

    def test_receiver_has_no_core_import_or_assert_based_validation(self):
        tree=ast.parse(VERIFY_PATH.read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        self.assertEqual(set(imports),{'fractions','hashlib','json','re'})
        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(tree)))

    def test_standalone_stdlib_process_without_package_or_producer(self):
        script="""import builtins, importlib.util, json, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('icde2027') or name == 'core':
        raise RuntimeError('producer import is forbidden')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('standalone_receiver', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request = json.loads(sys.argv[2]); certificate = json.loads(sys.argv[3])
result = module.verify_certificate(*request, certificate)
assert result['verified']
assert not any(n.startswith('icde2027') for n in sys.modules)
print(json.dumps(result, sort_keys=True))
"""
        request=lower_row_request();cert=solve(*request)
        result=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(VERIFY_PATH),
                               json.dumps(request),json.dumps(cert)],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout)['capacity'],'1/6')
        COUNTS['standalone_process_checks']+=1

    def test_receiver_does_not_mutate_or_alias_requests_and_certificates(self):
        request=lower_row_request();cert=solve(*request);before=copy.deepcopy((request,cert))
        result=receiver.verify_certificate(*request,cert)
        result['checked_fields'].clear();result['unchecked_fields'].clear()
        self.assertEqual((request,cert),before)

    def test_large_denominator_capacity_is_exact(self):
        epsilon=F(1,10**100+267)
        request=(raw_program([F(1,2),0],[F(1,2),F(1,2)-epsilon]),[2],[1],[[0,1]])
        self.assertEqual(F(receiver.verify_certificate(*request,solve(*request))['capacity']),epsilon)


class ExactOracleImplementationReview(unittest.TestCase):
    def test_467_registered_exact_epigraph_cases(self):
        self.assertEqual(hashlib.sha256((MATH_REVIEW/'exact_review.py').read_bytes()).hexdigest(),
                         'a1f5f2f69a91a94d1d9a24979d137980d31205e022b348245cfc7ab8c21cd0b2')
        self.assertEqual(hashlib.sha256((MATH_REVIEW/'FINAL_INDEX.json').read_bytes()).hexdigest(),
                         '4da66e77ffab062e5522cd1e8ae0f169fa0afa88411686bfdfed7093286a6bab')
        index=json.loads((MATH_REVIEW/'FINAL_INDEX.json').read_text())
        hashes={v['name']:v['sha256'] for v in index['files']}
        inputs=[];expected={}
        for filename in ('REGISTERED_CASES.json','ADDITIONAL_REGISTERED_CASES.json',
                         'CASE_RESULTS.json','ADDITIONAL_CASE_RESULTS.json'):
            path=MATH_REVIEW/filename
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),hashes[filename])
            records=json.loads(path.read_text())
            if 'REGISTERED' in filename:inputs.extend(records)
            else:expected.update({r['name']:F(r['value']) for r in records})
        self.assertEqual(len(inputs),467);self.assertEqual(len(expected),467)
        oracle=load_file_module(MATH_REVIEW/'exact_review.py','frozen_independent_fraction_oracle')
        for raw in inputs:
            with self.subTest(case=raw['name']):
                case={'name':raw['name'],'l':list(map(F,raw['l'])),'u':list(map(F,raw['u'])),
                      'p':list(map(F,raw['p'])),
                      'rows':[{'T':list(map(F,r['T'])),'a':F(r['a']),'b':F(r['b'])} for r in raw['rows']]}
                request=(raw_program(raw['l'],raw['u'],[(r['T'],r['a'],r['b']) for r in raw['rows']]),
                         [len(raw['l'])],[1],[raw['p']])
                exact=oracle.epigraph_oracle(case)
                cert=solve(*request)
                receipt=receiver.verify_certificate(*request,cert)
                self.assertEqual(F(cert['capacity']),exact['value'])
                self.assertEqual(exact['value'],expected[raw['name']])
                self.assertEqual(F(cert['contexts'][0]['cap']),exact['cap'])
                self.assertEqual(F(cert['contexts'][0]['reference_mean']),exact['capped_mean'])
                self.assertEqual(F(receipt['capacity']),exact['value'])
                COUNTS['exact_oracle_cases']+=1
                COUNTS['exact_oracle_systems']+=exact['systems']
                COUNTS['exact_oracle_vertices']+=len(exact['vertices'])
                ORACLE_RECORDS.append({'name':raw['name'],'oracle_capacity':str(exact['value']),
                                      'oracle_cap':str(exact['cap']), 'certificate':cert,'receipt':receipt})

    def test_30_multicontext_weighted_requests(self):
        box=(raw_program(['2/3',0],[1,'1/3']),[2],[1],[[0,1]])
        components=[(box,F(1,3)),(lower_row_request(),F(1,6)),(upper_row_request(),F(2,5))]
        weightsets=[[F(0),F(2,5),F(3,5)],[F(1,2),F(0),F(1,2)],
                    [F(1,3)]*3,[F(0),F(0),F(1)],[F(1),F(0),F(0)]]
        for perm in itertools.permutations(components):
            for weights in weightsets:
                request=merge_requests([r for r,v in perm],weights)
                cert=solve(*request);receipt=receiver.verify_certificate(*request,cert)
                expected=sum((w*v for w,(_,v) in zip(weights,perm)),F(0))
                self.assertEqual(F(receipt['capacity']),expected)
                COUNTS['multicontext_cases']+=1
                MULTICONTEXT_RECORDS.append({'expected_capacity':str(expected),
                                             'certificate':cert,'receipt':receipt})


if __name__=='__main__':
    unittest.main()
