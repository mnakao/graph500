/*
 * low_level_func.h
 *
 *  Created on: 2012/10/17
 *      Author: ueno
 */

#ifndef LOW_LEVEL_FUNC_H_
#define LOW_LEVEL_FUNC_H_

#include "parameters.h"

typedef struct FoldEdge {
	uint32_t v0_high;
	uint32_t v0_low;
	uint32_t v1;
} FoldEdge;

typedef struct FoldPacket {
	int num_edges;
#ifdef __cplusplus
	FoldEdge edges[BFS_PARAMS::PACKET_LENGTH];
#else
	FoldEdge edges[PACKET_LENGTH];
#endif
} FoldPacket;

#define SET_FOLDEDGE(e, v0_, v1_) \
	(e).v0_high = (v0_) >> 32;\
	(e).v0_low = (v0_);\
	(e).v1 = v1_;

#define GET_FOLDEDGE(e, v0_, v1_) \
	v0_ = ((int64_t)((e).v0_high) << 32) | (e).v0_low;\
	v1_ = (e).v1;

#ifdef __cplusplus
extern "C" {
#endif // #ifdef __cplusplus

#if LOW_LEVEL_FUNCTION

typedef struct RowPairData {
	int64_t bv_offset; // offset index
	int64_t v; // corresponding vertex
} RowPairData;

typedef struct BfsExtractContext {
	uint64_t* cq_bitmap;
	uint64_t* visited_bitmap;
	uint64_t* cq_summary; // forward
	uint64_t* visited_summary; // backward
	uint64_t* bv_bitmap;
	int64_t* bv_offsets;
	int64_t* bv_row_starts;
#if EDGES_IN_RAIL
	uint16_t *idx_array;
#else
	int32_t *idx_high;
	uint16_t *idx_low;
#endif
#if BIT_SCAN_TABLE
	uint8_t* bit_scan_table;
#endif
	int64_t local_verts_mask;
	int64_t v0_high_c; // forward
	int log_local_verts; // forward
	int log_r; // backward
	int64_t swizzed_r; // backward

	int64_t num_active_vertices; // backward
#if VERVOSE_MODE
	int64_t num_edge_relax;
	int64_t time_vertex_scan;
	int64_t time_core_proc;
#endif
	void (*submit_packetf)(void*, int dest_c); // forward
	void (*submit_packetb)(void*); // backward
	void* context;
	RowPairData *row_array;

	uint64_t cur_summary;
	int64_t summary_idx;
	int summary_bit_idx;
	int cq_offset; // forward
	int visited_offset; // backward
	int unvisited_count; // backward
} BfsExtractContext;

typedef struct BackwardRowState {
	int64_t row_end;
	int64_t word_idx;
	int64_t v1_local;
	int64_t c0;
	int64_t ri;
} BackwardRowState;

void bfs_forward_extract_edge(FoldPacket* packet_array, BfsExtractContext* p);
void bfs_backward_extract_edge(FoldPacket* packet, BfsExtractContext* p);

int64_t bfs_receiver_processing(
	const FoldEdge* const edges,
	const int num_edges,
	uint64_t* const visited,
	uint64_t* const nq_bitmap,
	int64_t* const pred,
	const int cur_level,
	const int log_local_verts,
	const int log_size,
	const int64_t local_verts_mask
	);

#endif // #if LOW_LEVEL_FUNCTION

#ifdef __cplusplus
} // extern "C" {
#endif // #ifdef __cplusplus

#endif /* LOW_LEVEL_FUNC_H_ */
