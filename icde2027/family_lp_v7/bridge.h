#ifndef FAMILY_LP_V7_BRIDGE_H
#define FAMILY_LP_V7_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Caller owns this zero-initialized record; payload is released by flp_release.
 * outcome: 0 exact optimum, 1 invalid request, 2 solver refusal, 3 exception.
 * All durations include failed paths. The input/output wire is documented in README.
 */
typedef struct flp_result_v1 {
    uint32_t abi_version;
    int32_t outcome;
    int32_t solver_status;
    uint64_t model_ns;
    uint64_t solve_ns;
    uint64_t export_ns;
    uint64_t total_ns;
    char* payload;
    size_t payload_size;
} flp_result_v1;
/* solver_limit_ns=0 keeps the backend default; a nonzero value limits optimize()
 * only. It is not an outer Python service timeout or memory limit. */
int flp_solve_v1(const char* request, size_t request_size, uint64_t solver_limit_ns, flp_result_v1* result);
void flp_release_v1(flp_result_v1* result);
const char* flp_build_info_v1(void);
size_t flp_result_size_v1(void);
#ifdef __cplusplus
}
#endif
#endif
