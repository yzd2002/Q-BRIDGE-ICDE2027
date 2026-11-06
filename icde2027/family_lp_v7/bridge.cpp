#include "bridge.h"
#include "soplex.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(SOPLEX_WITH_GMP) || !defined(SOPLEX_WITH_BOOST) || !defined(SOPLEX_WITH_MPFR)
#error "The reviewed exact GMP/Boost/MPFR SoPlex build is required"
#endif

namespace {
using Q=soplex::Rational;
using Clock=std::chrono::steady_clock;
struct Invalid:std::runtime_error { using std::runtime_error::runtime_error; };
struct Refusal:std::runtime_error { using std::runtime_error::runtime_error; };
struct Bound { bool finite; Q value; };
struct Column { Q objective; Bound lower,upper; };
struct Row { Bound lower,upper; std::vector<std::pair<int,Q>> values; };

void need(bool condition,const char* message) { if(!condition)throw Invalid(message); }
void certify(bool condition,const char* message) { if(!condition)throw Refusal(message); }
uint64_t ns(Clock::time_point from,Clock::time_point to) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to-from).count());
}
std::string render(const Q& q) {
    std::ostringstream out;out.imbue(std::locale::classic());out<<numerator(q)<<'/'<<denominator(q);return out.str();
}
std::string token(std::istringstream& input) {
    std::string value;need(static_cast<bool>(input>>value),"Missing input token");return value;
}
bool decimal_integer(const std::string& s,bool signed_value) {
    size_t i=signed_value && !s.empty() && s[0]=='-' ? 1 : 0;
    if(i==s.size())return false;
    if(s[i]=='0' && i+1!=s.size())return false;
    for(;i<s.size();++i)if(s[i]<'0'||s[i]>'9')return false;
    return true;
}
int count(std::istringstream& input) {
    const auto value=token(input);need(decimal_integer(value,false),"Noncanonical dimension/index");
    uint64_t result=0;
    for(char ch:value) { result=result*10+static_cast<unsigned>(ch-'0');need(result<=INT_MAX,"Dimension/index exceeds native int"); }
    return static_cast<int>(result);
}
Q rational(const std::string& text) {
    const auto slash=text.find('/');
    need(slash!=std::string::npos && text.find('/',slash+1)==std::string::npos,"Rational must be canonical n/d");
    auto n=text.substr(0,slash),d=text.substr(slash+1);
    need(decimal_integer(n,true)&&decimal_integer(d,false)&&d!="0","Invalid exact rational token");
    Q result(text.c_str());
    // Boost 1.73's GMP string assignment calls mpq_set_str without reducing.
    // Explicitly reduce before comparing the supplied canonical wire token.
    mpq_canonicalize(result.backend().data());
    need(render(result)==text,"Noncanonical exact rational token");return result;
}
Bound bound(std::istringstream& input) {
    const auto value=token(input);return value=="*"?Bound{false,Q(0)}:Bound{true,rational(value)};
}
void interval(const Bound& lower,const Bound& upper) {
    need(!lower.finite||!upper.finite||lower.value<=upper.value,"Reversed interval");
    // SoPlex uses a finite sentinel for infinity. Never silently turn an
    // explicitly finite input bound into an unbounded side.
    const Q infinity(soplex::infinity);
    need(!lower.finite || (-infinity<lower.value && lower.value<infinity),"Finite lower bound collides with SoPlex infinity");
    need(!upper.finite || (-infinity<upper.value && upper.value<infinity),"Finite upper bound collides with SoPlex infinity");
}
std::string quote(const std::string& text) {
    std::string result="\"";
    for(unsigned char ch:text) {
        if(ch=='\"'||ch=='\\') { result+='\\';result+=static_cast<char>(ch); }
        else if(ch=='\n')result+="\\n";
        else if(ch>=32 && ch<=126)result+=static_cast<char>(ch);
        else result+='?';
    }
    return result+'\"';
}
template<class V> std::string vector_json(const V& values,int size) {
    std::string out="[";
    for(int i=0;i<size;++i) { if(i)out+=',';out+=quote(render(values[i])); }
    return out+"]";
}
std::string model_roundtrip(const soplex::SoPlex& solver,const std::vector<Column>& columns,const std::vector<Row>& rows) {
    certify(solver.numColsRational()==static_cast<int>(columns.size()) &&
            solver.numRowsRational()==static_cast<int>(rows.size()),"Imported model dimensions changed");
    const Q infinity(soplex::infinity);std::ostringstream out;out.imbue(std::locale::classic());
    out<<"FAMILY_LP_EXACT/1\n"<<columns.size()<<' '<<rows.size()<<'\n';
    auto encode_bound=[&](const Bound& wanted,const Q& observed,bool lower) {
        certify(observed==(wanted.finite?wanted.value:(lower?-infinity:infinity)),"Imported rational bound changed");
        return wanted.finite?render(observed):std::string("*");
    };
    for(int i=0;i<static_cast<int>(columns.size());++i) {
        const auto& c=columns[i];const Q objective=solver.objRational(i);
        certify(objective==c.objective,"Imported rational objective changed");
        out<<render(objective)<<' '<<encode_bound(c.lower,solver.lowerRational(i),true)
           <<' '<<encode_bound(c.upper,solver.upperRational(i),false)<<'\n';
    }
    for(int j=0;j<static_cast<int>(rows.size());++j) {
        const auto& row=rows[j];const auto& observed=solver.rowVectorRational(j);
        std::vector<std::pair<int,Q>> values;values.reserve(observed.size());
        for(int k=0;k<observed.size();++k)values.emplace_back(observed.index(k),observed.value(k));
        std::sort(values.begin(),values.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        certify(values==row.values,"Imported original rational coefficient changed");
        out<<encode_bound(row.lower,solver.lhsRational(j),true)<<' '
           <<encode_bound(row.upper,solver.rhsRational(j),false)<<' '<<values.size();
        for(const auto& entry:values)out<<' '<<entry.first<<' '<<render(entry.second);
        out<<'\n';
    }
    return out.str();
}
void payload(flp_result_v1* result,const std::string& text) {
    auto* buffer=static_cast<char*>(std::malloc(text.size()+1));
    if(!buffer)throw std::bad_alloc();
    std::memcpy(buffer,text.data(),text.size());buffer[text.size()]='\0';
    result->payload=buffer;result->payload_size=text.size();
}
}

extern "C" const char* flp_build_info_v1(void) {
    return "family-lp-bridge/1;SoPlex=8.1.0;commit=a00610534e044f35aa298059eb963312336c4c9c;GMP;Boost;MPFR;exact-rational";
}

extern "C" void flp_release_v1(flp_result_v1* result) {
    if(result) { std::free(result->payload);result->payload=nullptr;result->payload_size=0; }
}

extern "C" size_t flp_result_size_v1(void) { return sizeof(flp_result_v1); }

extern "C" int flp_solve_v1(const char* request,size_t request_size,uint64_t solver_limit_ns,flp_result_v1* result) {
    if(!result)return 1;
    *result={};result->abi_version=1;result->outcome=3;result->solver_status=INT_MIN;
    const auto started=Clock::now();auto phase_started=started;int phase=0;
    auto close_phase=[&]() {
        const auto current=Clock::now();const auto elapsed=ns(phase_started,current);
        if(phase==0)result->model_ns+=elapsed;
        else if(phase==1)result->solve_ns+=elapsed;
        else result->export_ns+=elapsed;
        phase_started=current;
    };
    try {
        need(request && request_size>0,"Null or empty request");
        std::istringstream input(std::string(request,request_size));
        need(token(input)=="FAMILY_LP_EXACT/1","Unsupported native wire schema");
        const int n=count(input),m=count(input);
        need(n>0 && static_cast<size_t>(n)<=request_size && static_cast<size_t>(m)<=request_size,
             "Dimension is empty or inconsistent with payload size");
        std::vector<Column> columns;columns.reserve(n);
        std::vector<Row> rows;rows.reserve(m);
        for(int i=0;i<n;++i) {
            Column c{rational(token(input)),bound(input),bound(input)};
            interval(c.lower,c.upper);columns.push_back(std::move(c));
        }
        for(int j=0;j<m;++j) {
            Row row{bound(input),bound(input),{}};interval(row.lower,row.upper);
            int nonzero=count(input);need(nonzero<=n,"Row nonzero count exceeds dimension");
            row.values.reserve(nonzero);int previous=-1;
            for(int k=0;k<nonzero;++k) {
                int i=count(input);Q value=rational(token(input));
                need(i>previous && i<n && value!=0,"Unsorted/duplicate/out-of-range/zero sparse entry");
                previous=i;row.values.emplace_back(i,std::move(value));
            }
            rows.push_back(std::move(row));
        }
        std::string extra;need(!(input>>extra),"Trailing input token");
        soplex::SoPlex solver;
        need(solver.setIntParam(soplex::SoPlex::READMODE,soplex::SoPlex::READMODE_RATIONAL),"READMODE refused");
        need(solver.setIntParam(soplex::SoPlex::SOLVEMODE,soplex::SoPlex::SOLVEMODE_RATIONAL),"SOLVEMODE refused");
        need(solver.setIntParam(soplex::SoPlex::CHECKMODE,soplex::SoPlex::CHECKMODE_RATIONAL),"CHECKMODE refused");
        need(solver.setIntParam(soplex::SoPlex::SYNCMODE,soplex::SoPlex::SYNCMODE_AUTO),"SYNCMODE refused");
        need(solver.setRealParam(soplex::SoPlex::FEASTOL,0.0),"FEASTOL refused");
        need(solver.setRealParam(soplex::SoPlex::OPTTOL,0.0),"OPTTOL refused");
        need(solver.setIntParam(soplex::SoPlex::OBJSENSE,soplex::SoPlex::OBJSENSE_MINIMIZE),"Objective sense refused");
        need(solver.setIntParam(soplex::SoPlex::VERBOSITY,soplex::SoPlex::VERBOSITY_ERROR),"Verbosity refused");
        if(solver_limit_ns)need(solver.setRealParam(soplex::SoPlex::TIMELIMIT,
            static_cast<double>(solver_limit_ns)/1e9),"Native optimize time limit refused");
        const Q infinity(soplex::infinity);soplex::DSVectorRational empty(0);
        for(const auto& c:columns)solver.addColRational(soplex::LPColRational(c.objective,empty,
            c.upper.finite?c.upper.value:infinity,c.lower.finite?c.lower.value:-infinity));
        for(const auto& r:rows) {
            soplex::DSVectorRational values(static_cast<int>(r.values.size()));
            for(const auto& entry:r.values)values.add(entry.first,entry.second);
            solver.addRowRational(soplex::LPRowRational(r.lower.finite?r.lower.value:-infinity,
                values,r.upper.finite?r.upper.value:infinity));
        }
        const auto imported_wire=model_roundtrip(solver,columns,rows);
        close_phase();phase=1;
        const auto status=solver.optimize();result->solver_status=static_cast<int>(status);
        close_phase();phase=2;
        certify(status==soplex::SPxSolver::OPTIMAL,"SoPlex did not return OPTIMAL");
        certify(solver.isPrimalFeasible() && solver.isDualFeasible(),"SoPlex exact feasibility flags missing");
        soplex::DVectorRational point(n),dual(m),reduced(n);
        certify(solver.getPrimalRational(point) && solver.getDualRational(dual) && solver.getRedCostRational(reduced),
                "Exact rational primal/dual/reduced-cost export failed");
        Q primal=0,lower_bound=0;std::vector<Q> z;z.reserve(n);
        for(int i=0;i<n;++i) {
            const auto& c=columns[i];
            certify((!c.lower.finite||c.lower.value<=point[i])&&(!c.upper.finite||point[i]<=c.upper.value),"Exact original column infeasible");
            primal+=c.objective*point[i];z.push_back(c.objective);
        }
        for(int j=0;j<m;++j) {
            const auto& r=rows[j];Q value=0;
            for(const auto& e:r.values) { value+=e.second*point[e.first];z[e.first]-=dual[j]*e.second; }
            certify((!r.lower.finite||r.lower.value<=value)&&(!r.upper.finite||value<=r.upper.value),"Exact original row infeasible");
            if(dual[j]>0) { certify(r.lower.finite,"Positive row dual on free lower side");lower_bound+=dual[j]*r.lower.value; }
            if(dual[j]<0) { certify(r.upper.finite,"Negative row dual on free upper side");lower_bound+=dual[j]*r.upper.value; }
        }
        for(int i=0;i<n;++i) {
            const auto& c=columns[i];certify(z[i]==reduced[i],"Exact reduced-cost sign/stationarity mismatch");
            if(z[i]>0) { certify(c.lower.finite,"Positive reduced cost with free lower bound");lower_bound+=z[i]*c.lower.value; }
            if(z[i]<0) { certify(c.upper.finite,"Negative reduced cost with free upper bound");lower_bound+=z[i]*c.upper.value; }
        }
        certify(primal==lower_bound && primal==solver.objValueRational(),"Exact original primal-dual objective equality failed");
        std::string out="{\"schema_version\":\"family-lp-native-result/1\",\"status\":\"exact_optimal\",\"objective\":"+quote(render(primal));
        out+=",\"primal\":"+vector_json(point,n)+",\"row_duals\":"+vector_json(dual,m)+",\"reduced_costs\":"+vector_json(reduced,n);
        out+=",\"model_wire_roundtrip\":"+quote(imported_wire)+",\"exact_original_primal_dual_equality\":true}";
        payload(result,out);result->outcome=0;
    } catch(const Invalid& error) {
        close_phase();phase=2;
        result->outcome=1;
        try { payload(result,"{\"status\":\"invalid_input\",\"error\":"+quote(error.what())+"}"); } catch(...) {}
    } catch(const Refusal& error) {
        close_phase();phase=2;
        result->outcome=2;
        try { payload(result,"{\"status\":\"solver_refusal\",\"error\":"+quote(error.what())+"}"); } catch(...) {}
    } catch(const std::exception& error) {
        close_phase();phase=2;
        result->outcome=3;
        try { payload(result,"{\"status\":\"native_exception\",\"error\":"+quote(error.what())+"}"); } catch(...) {}
    } catch(...) {
        close_phase();phase=2;
        result->outcome=3;
        try { payload(result,"{\"status\":\"native_exception\",\"error\":\"Unknown native exception\"}"); } catch(...) {}
    }
    close_phase();result->total_ns=ns(started,Clock::now());return result->outcome;
}
