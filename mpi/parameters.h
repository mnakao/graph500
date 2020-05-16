#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#define PRINT_WITH_TIME 1
#define TOP_DOWN_SEND_LB 2
#define TOP_DOWN_RECV_LB 1
#define ISOLATE_FIRST_EDGE 1
#define CONSOLIDATE_IFE_PROC 1
#define DENOM_TOPDOWN_TO_BOTTOMUP 15000.0
#define DEMON_BOTTOMUP_TO_TOPDOWN 8.0
#define DENOM_BITMAP_TO_LIST 2.0 // temp
#define LOW_LEVEL_FUNCTION 1
#define STREAM_UPDATE 1
#define CACHE_LINE 256
#define PAGE_SIZE 65536
#define IMD_OUT stderr

typedef uint64_t BitmapType;
typedef uint64_t TwodVertex;
typedef uint32_t LocalVertex;

#ifdef __cplusplus
namespace PRM { //
#endif // #ifdef __cplusplus

#define SIZE_OF_SUMMARY_IS_EQUAL_TO_WARP_SIZE

enum {
	NUM_BFS_ROOTS = 16,
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
