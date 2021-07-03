/*
 * parameters.h
 *
 *  Created on: Mar 2, 2012
 *      Author: koji
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

// for the systems that contains NUMA nodes
#define NUMA_BIND 0
#define SHARED_MEMORY 0
#define CPU_BIND_CHECK 0
#define PRINT_BINDING 0

// Switching the task assignment for the main thread and the sub thread
// 0: MPI is single mode: Main -> MPI, Sub: OpenMP
// 1: MPI is funneled mode: Main -> OpenMP, Sub: MPI
// Since communication and computation is overlapped, we cannot have main thread do both tasks.
#define MPI_FUNNELED 1
#define OPENMP_SUB_THREAD 0

// Validation Level: 0: No validation, 1: validate at first time only, 2: validate all results
// Note: To conform to the specification, you must set 2
#define VALIDATION_LEVEL 2

// General Settings
#define PRINT_WITH_TIME 1
#ifndef VERVOSE_MODE
#define VERVOSE_MODE 1
#endif

// General Optimizations
// 0: completely off, 1: only reduce isolated vertices, 2: sort by degree and reduce isolated vertices
#ifndef VERTEX_REORDERING
#define VERTEX_REORDERING 2
#endif
// 0: put all edges to temporally buffer, 1: count first, 2: hybrid
#define TOP_DOWN_SEND_LB 2
#define TOP_DOWN_RECV_LB 1
#define BOTTOM_UP_OVERLAP_PFS 1

// for Fugaku
#define ENABLE_UTOFU 0
#define FUGAKU_MPI_PRINT_STATS 0
#define ENABLE_INLINE_ATOMICS 0

#define BFELL 0

// Optimization for CSR
#define ISOLATE_FIRST_EDGE 1
#define DEGREE_ORDER 0
#define DEGREE_ORDER_ONLY_IE 0
#define CONSOLIDATE_IFE_PROC 1

// We omit initialize predecessor array when this option is enabled.
// WARNING: In the most case, BFS generates correct answer without initializing predecessor array
// because all the vertexes reached in the previous would be reached in the current run.
// But this is not true in the general case. BFS may generate wrong answer in some situation.
#define INIT_PRED_ONCE 0

#define PRE_EXEC_TIME 0 // 0 or 600 seconds

// org = 1000
//#define DENOM_TOPDOWN_TO_BOTTOMUP 2000.0
#define DENOM_TOPDOWN_TO_BOTTOMUP 15000.0
#define DEMON_BOTTOMUP_TO_TOPDOWN 8.0
#define DENOM_BITMAP_TO_LIST 2.0 // temp

// atomic level of scanning Shared Visited
// 0: no atomic operation
// 1: non atomic read and atomic write
// 2: atomic read-write
#define SV_ATOMIC_LEVEL 0

#define SIMPLE_FOLD_COMM 1
#define KD_PRINT 0

#define NETWORK_PROBLEM_AYALISYS 0
#define WITH_VALGRIND 0

#define SGI_OMPLACE_BUG 0

#ifdef __FUJITSU
# define LOW_LEVEL_FUNCTION 1
# define STREAM_UPDATE 1
# define ENABLE_FJMPI 0
#else
# define ENABLE_FJMPI 0
#endif

#if BFELL
#	undef ISOLATE_FIRST_EDGE
#	define ISOLATE_FIRST_EDGE 0
#	undef DEGREE_ORDER
#	define DEGREE_ORDER 0
#endif

#ifndef _OPENMP
// turn OFF when OpenMP is not enabled
#	undef OPENMP_SUB_THREAD
#	define OPENMP_SUB_THREAD 0
#endif
#ifdef __FUJITSU
#define CACHE_LINE 256
#define PAGE_SIZE 65536
#else
#define CACHE_LINE 128
#define PAGE_SIZE 8192
#endif

#define IMD_OUT stderr

typedef uint8_t SortIdx;
typedef uint64_t BitmapType;
typedef uint64_t TwodVertex;
typedef uint32_t LocalVertex;

#ifdef __cplusplus
namespace PRM { //
#endif // #ifdef __cplusplus

#define SIZE_OF_SUMMARY_IS_EQUAL_TO_WARP_SIZE

enum {
#ifdef REAL_BENCHMARK
	NUM_BFS_ROOTS = 64, // spec: 64
#else
	NUM_BFS_ROOTS = 16,
#endif
	PACKET_LENGTH = 1024,
	COMM_BUFFER_SIZE = 32*1024, // !!IMPORTANT VALUE!!
	PRE_ALLOCATE_COMM_BUFFER = 14,
	SEND_BUFFER_LIMIT = 6,
	TOP_DOWN_PENDING_WIDTH = 2000,
	BOTTOM_UP_BUFFER = 16,
	PREFETCH_DIST = 16,
	LOG_BIT_SCAN_TABLE = 11,
	LOG_NBPE = 6,
	LOG_BFELL_SORT = 8,
	NUM_BOTTOM_UP_STREAMS = 4,

	// non-parameters
	NBPE = 1 << LOG_NBPE, // <= sizeof(BitmapType)*8
	NBPE_MASK = NBPE - 1,
	BFELL_SORT = 1 << LOG_BFELL_SORT,
	BFELL_SORT_MASK = BFELL_SORT - 1,

	USERSEED1 = 2,
	USERSEED2 = 3,

	TOP_DOWN_FOLD_TAG = 0,
	BOTTOM_UP_WAVE_TAG = 1,
	BOTTOM_UP_PRED_TAG = 2,
	MY_EXPAND_TAG1 = 3,
	MY_EXPAND_TAG2 = 4,
};

#ifdef __cplusplus
} // namespace PRM {

#endif // #ifdef __cplusplus

#endif /* PARAMETERS_H_ */
