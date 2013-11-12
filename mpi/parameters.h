/*
 * parameters.h
 *
 *  Created on: Mar 2, 2012
 *      Author: koji
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

// Project includes
#define VERVOSE_MODE 1
#define PROFILING_MODE 1
#define DETAILED_PROF_MODE 0

#define BFELL 1
#define STREAM_UPDATE 1
#define BF_DEEPER_ASYNC 1
#define SHARED_MEM_VISITED 1

#define SHARED_VISITED_OPT 1

#define FAKE_VERTEX_SORTING 0 // benchmark: 0
#define FAKE_VISITED_SHARING 0 // benchmark: 0

#define PRE_EXEC_TIME 120 // 300 seconds

#define BFS_BACKWARD 1
#define VLQ_COMPRESSION 0
#define BFS_EXPAND_COMPRESSION 1
#define VERTEX_SORTING 0 // do not support backward search
#define BFS_FORWARD_PREFETCH 1 // for FUJITSU
#define BFS_BACKWARD_PREFETCH 1
#define GRAPH_BITVECTOR 1
#define GRAPH_BITVECTOR_OFFSET 1
#define LOW_LEVEL_FUNCTION 0
#define BIT_SCAN_TABLE 0
#define AVOID_BUSY_WAIT 0
#define EDGES_IN_RAIL 1

#define PREFETCH_INST_WEIGHTED 0

// set 0 with CUDA
#define SHARED_VISITED_STRIPE 0

// Validation Level: 0: No validation, 1: validate at first time only, 2: validate all results
// Note: To conform to the specification, you must set 2
#define VALIDATION_LEVEL 2

#define CUDA_ENABLED 0
#define CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE 1
#define CUDA_CHECK_PRINT_RANK

// atomic level of scanning Shared Visited
// 0: no atomic operation
// 1: non atomic read and atomic write
// 2: atomic read-write
#define SV_ATOMIC_LEVEL 0

#define SIMPLE_FOLD_COMM 1
#define DEBUG_PRINT 0
#define KD_PRINT 0

#define DISABLE_CUDA_CONCCURENT 0
#define NETWORK_PROBLEM_AYALISYS 0

#ifdef __FUJITSU

#define SWITCH_FUJI_PROF 0
#define ESCAPE_ATOMIC_FUNC 1
#define USE_SPARC_ASM_POPC 1
#define FCC_OMP_ASM_BUG 1
//#define AVOID_BUSY_WAIT 1

#else // #ifdef __FUJITSU

#define SWITCH_FUJI_PROF 0
#define ESCAPE_ATOMIC_FUNC 0
#define USE_SPARC_ASM_POPC 0
#define FCC_OMP_ASM_BUG 0
//#define AVOID_BUSY_WAIT 0

#endif // #ifdef __FUJITSU

#define CACHE_LINE 128
#define PAGE_SIZE 4096

#define IMD_OUT stderr

typedef uint8_t SortIdx;
typedef uint64_t BitmapType;
typedef uint32_t TwodVertex;

#ifdef __cplusplus
namespace PRM { //
#endif // #ifdef __cplusplus

#define SIZE_OF_SUMMARY_IS_EQUAL_TO_WARP_SIZE

enum {
	USERSEED1 = 2,
	USERSEED2 = 3,
	NUM_BFS_ROOTS = 64, // spec: 64
#if CUDA_ENABLED
	PACKET_LENGTH = 256,
	LOG_PACKET_LENGTH = 8,
#else
	PACKET_LENGTH = 1024,
#endif
	COMM_BUFFER_SIZE = 32*1024, // !!IMPORTANT VALUE!!
	PRE_ALLOCATE_COMM_BUFFER = 14,
	SEND_BUFFER_LIMIT = 6,

	DEMON_TOPDOWN_TO_BOTTOMUP = 1000,
	DEMON_BOTTOMUP_TO_TOPDOWN = 2,

	BOTTOM_UP_BUFFER = 8,

	TOP_DOWN_FOLD_TAG = 0,
	BOTTOM_UP_WAVE_TAG = 1,
	BOTTOM_UP_PRED_TAG = 2,

	PREFETCH_DIST = 16,

	LOG_BIT_SCAN_TABLE = 11,
	LOG_NBPE = 6,
	LOG_BFELL_SORT = 8,

	// non-parameters
	NBPE = 1 << LOG_NBPE, // <= sizeof(BitmapType)*8
	NBPE_MASK = NBPE - 1,
	BFELL_SORT = 1 << LOG_BFELL_SORT,
	BFELL_SORT_MASK = BFELL_SORT - 1,
	BIT_SCAN_TABLE_SIZE = 1 << LOG_BIT_SCAN_TABLE,
	BIT_SCAN_TABLE_MASK = BIT_SCAN_TABLE_SIZE - 1,
};

#ifdef __cplusplus
} // namespace PRM {

namespace GPU_PARAMS {
enum {
	LOG_WARP_SIZE = 5, // 32
	LOG_WARPS_PER_BLOCK = 3, // 8
	LOOPS_PER_THREAD = 8,
	ACTIVE_THREAD_BLOCKS = 64,
	NUMBER_CUDA_STREAMS = 4,

	// Dynamic Scheduling
	DS_TILE_SCALE = 16,
	DS_BLOCK_SCALE = 8,
	BLOCKS_PER_LAUNCH_RATE = 8*1024, // for debug

	EXPAND_DECODE_BLOCK_LENGTH = 16*1024*1024, // 64MB
	EXPAND_STREAM_BLOCK_LENGTH = EXPAND_DECODE_BLOCK_LENGTH * 2, // 32MB
	EXPAND_PACKET_LIST_LENGTH = EXPAND_DECODE_BLOCK_LENGTH / PRM::PACKET_LENGTH * 2,

	LOG_PACKING_EDGE_LISTS = 5, // 2^5 = 32 // for debug
	LOG_CQ_SUMMARIZING = LOG_WARP_SIZE, // 32 ( number of threads in a warp )

	// non-parameters
	WARP_SIZE = 1 << LOG_WARP_SIZE,
	WARPS_PER_BLOCK = 1 << LOG_WARPS_PER_BLOCK,
	LOG_THREADS_PER_BLOCK = LOG_WARP_SIZE + LOG_WARPS_PER_BLOCK,
	THREADS_PER_BLOCK = WARP_SIZE*WARPS_PER_BLOCK,
	// Number of max threads launched by 1 GPU kernel.
	MAX_ACTIVE_WARPS = WARPS_PER_BLOCK*ACTIVE_THREAD_BLOCKS,
	TEMP_BUFFER_LINES = MAX_ACTIVE_WARPS*LOOPS_PER_THREAD,

	READ_GRAPH_OUTBUF_SIZE = THREADS_PER_BLOCK * BLOCKS_PER_LAUNCH_RATE * 2,
//	READ_GRAPH_OUTBUF_SIZE = THREADS_PER_BLOCK * BLOCKS_PER_LAUNCH_RATE / 2,

	NUMBER_PACKING_EDGE_LISTS = (1 << LOG_PACKING_EDGE_LISTS),
	NUMBER_CQ_SUMMARIZING = (1 << LOG_CQ_SUMMARIZING),
};
} // namespace GPU_PARAMS {
#endif // #ifdef __cplusplus

#endif /* PARAMETERS_H_ */
