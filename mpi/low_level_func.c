/*
 * low_level_func.c
 *
 *  Created on: 2012/10/17
 *      Author: ueno
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/time.h>

#include "low_level_func.h"

#ifdef __sparc_v9__

inline int64_t atomic_add_64_(int64_t* ptr, int64_t val) {
	int64_t old_value;
	__asm__ (
			"ldx     [%2], %0\n"
	"1:\n\t"
			"add    %0, %3, %%l0\n\t"
			"casx   [%2], %0, %%l0\n\t"
			"cmp    %0, %%l0\n\t"
			"bne,a,pn %%xcc, 1b\n\t"
			"mov    %%l0, %0\n\t"
			:"=&r"(old_value),"=m"(*ptr)
			:"r"(ptr),"r"(val)
			:"%l0","cc"
			);
	return old_value;
}

inline int32_t atomic_add_32_(int32_t* ptr, int32_t val) {
	int32_t old_value;
	__asm__ (
			"ldsw   [%2], %0\n"
	"1:\n\t"
			"add    %0, %3, %%l0\n\t"
			"cas    [%2], %0, %%l0\n\t"
			"cmp    %0, %%l0\n\t"
			"bne,a,pn %%icc, 1b\n\t"
			"mov    %%l0, %0\n\t"
			:"=&r"(old_value),"=m"(*ptr)
			:"r"(ptr),"r"(val)
			:"%l0","cc"
			);
	return old_value;
}

inline uint64_t atomic_or_64_(uint64_t* ptr, uint64_t val) {
	uint64_t old_value;
	__asm__ (
			"ldx     [%2], %0\n"
	"1:\n\t"
			"or     %0, %3, %%l0\n\t"
			"casx   [%2], %0, %%l0\n\t"
			"cmp    %0, %%l0\n\t"
			"bne,a,pn %%xcc, 1b\n\t"
			"mov    %%l0, %0\n\t"
			:"=&r"(old_value),"=m"(*ptr)
			:"r"(ptr),"r"(val)
			:"%l0","cc"
			);
	return old_value;
}

inline int popcount_64_(uint64_t n) {
	int c;
	__asm__(
			"popc %1, %0\n\t"
			:"=r"(c)
			:"r"(n)
			);
	return c;
}

#else // #ifdef __sparc_v9__

#define atomic_add_64_ __sync_fetch_and_add
#define atomic_or_64_ __sync_fetch_and_or
#define popcount_64_ __builtin_popcountl

#endif // #ifdef __sparc_v9__

enum {
	LOG_PACKING_EDGE_LISTS = 6, // 2^6 = 64
	LOG_CQ_SUMMARIZING = 4, // 2^4 = 16 -> sizeof(int64_t)*16 = 128bytes
	NUMBER_PACKING_EDGE_LISTS = (1 << LOG_PACKING_EDGE_LISTS),
	NUMBER_CQ_SUMMARIZING = (1 << LOG_CQ_SUMMARIZING),
};
typedef uint64_t BitmapType;

int64_t get_time_in_microsecond()
{
	struct timeval l;
	gettimeofday(&l, NULL);
	return ((int64_t)l.tv_sec*1000000 + l.tv_usec);
}

int bfs_forward_fill_new_rows(BfsExtractContext* p)
{
#if PROFILING_MODE
	int64_t tk_vscan = get_time_in_microsecond();
#endif
	BitmapType* cq_bitmap = p->cq_bitmap;
	BitmapType* bv_bitmap = p->bv_bitmap;
	int64_t* bv_offsets = p->bv_offsets;
	uint8_t* bit_scan_table = p->bit_scan_table;
	const int64_t v0_high_c = p->v0_high_c;

	RowPairData* row_array = p->row_array;

	BitmapType cur_summary = p->cur_summary;
	int64_t summary_idx = p->summary_idx;
	int summary_bit_idx = p->summary_bit_idx;

	int offset = 0;
	for( ; summary_bit_idx < 64; ++summary_bit_idx) {
		if(cur_summary & ((BitmapType)1 << summary_bit_idx)) {
			int cq_offset = p->cq_offset;
			int64_t cq_base = ((int64_t)summary_idx*64 + summary_bit_idx)*NUMBER_CQ_SUMMARIZING;

			// prefetch
		//	__builtin_prefetch(&cq_bitmap[cq_base + NUMBER_CQ_SUMMARIZING], 0);
		//	__builtin_prefetch(&bv_bitmap[cq_base + NUMBER_CQ_SUMMARIZING], 0);
		//	__builtin_prefetch(&bv_offsets[cq_base + NUMBER_CQ_SUMMARIZING], 0);

			for( ; cq_offset < NUMBER_CQ_SUMMARIZING; ++cq_offset) {
				int64_t e0 = cq_base + cq_offset;
				BitmapType bitmap_k = cq_bitmap[e0] & bv_bitmap[e0];
				// write zero after read
				cq_bitmap[e0] = 0;
				if(bitmap_k == 0) continue;

				const int64_t v0_high = (e0 << LOG_PACKING_EDGE_LISTS) | v0_high_c;
				const BitmapType bv_bits = bv_bitmap[e0];
				const int64_t bv_offset_base = bv_offsets[e0];

				int row_lowbits = -1;
				while(bitmap_k) {
					int scan_bits = bitmap_k & BIT_SCAN_TABLE_MASK;
					int lsb = bit_scan_table[scan_bits];
					bitmap_k >>= lsb;
					row_lowbits += lsb;
					if(scan_bits) {
						const BitmapType bv = (bv_bits & (((BitmapType)1 << row_lowbits) - 1));
						const int64_t bv_offset = bv_offset_base + popcount_64_(bv);
						// FCC will insert software prefetch here.
						row_array[offset].bv_offset = bv_offset;
						row_array[offset].v = v0_high | row_lowbits;
						++offset;
					}
				}

				if(offset > BNL_ARRAY_FILL_LENGTH) {
					p->summary_bit_idx = summary_bit_idx;
					p->cq_offset = cq_offset;
#if PROFILING_MODE
					p->time_vertex_scan += get_time_in_microsecond() - tk_vscan;;
#endif
					return offset;
				}
			}
			p->cq_offset = 0;
		}
	}
	p->summary_bit_idx = summary_bit_idx;
#if PROFILING_MODE
	p->time_vertex_scan += get_time_in_microsecond() - tk_vscan;;
#endif
	return offset;
}

void bfs_forward_extract_edge(FoldPacket* packet_array, BfsExtractContext* p)
{
	BitmapType* shared_visited = p->visited_bitmap;
#if EDGES_IN_RAIL
	uint16_t *idx_array = p->idx_array;
#else
	int32_t *idx_high = p->idx_high;
	uint16_t *idx_low = p->idx_low;
#endif
	const int64_t local_verts_mask = p->local_verts_mask;
	const int64_t log_local_verts = p->log_local_verts;
	void (*submit_packet)(void*, int) = p->submit_packetf;
	void* context = p->context;
#if VERVOSE_MODE
	int64_t num_edge_relax = 0;
#endif

	struct {
		int64_t v0, word_idx, c1;
	} tmp_edges[PREFETCH_DIST]; // 24 * 8 = 192bytes (<= 2 cache line)

	int new_row_idx = 0;

	// initialize filling new rows state
	p->summary_bit_idx = 0;
	p->cq_offset = 0;


	// fill new rows
	int num_new_rows = bfs_forward_fill_new_rows(p);

	if(num_new_rows == 0) return ;

	// TODO: prefetch row_array

	int64_t bv_offset = p->row_array[new_row_idx].bv_offset;
	int64_t ri = p->bv_row_starts[bv_offset];
	int64_t row_end = p->bv_row_starts[bv_offset + 1];
	int64_t cur_v0_local = p->row_array[new_row_idx].v;
	++new_row_idx;

	int active_edges = PREFETCH_DIST;
	do {
		int i = 0;

		// prefetch packet, TODO: performance evaluation
	//	__builtin_prefetch(&packet->v0_list[pk_num_edges], 1);
	//	__builtin_prefetch(&packet->v1_list[pk_num_edges], 1);

		do {
			int end;
			if(PREFETCH_DIST < i + row_end - ri) {
				end = PREFETCH_DIST;
	//			__builtin_prefetch(&p->idx_high[ri + PREFETCH_DIST*2], 0);
	//			__builtin_prefetch(&p->idx_low[ri + PREFETCH_DIST*2], 0);
			}
			else {
				end = i + row_end - ri;
			}
#ifdef __FUJITSU
//#pragma loop unroll 8
#endif
			for( ; i < end; ++i, ++ri) {			// read access
#if EDGES_IN_RAIL
				const int64_t raw = (int64_t)idx_array[ri*3 + 0] |
									(int64_t)idx_array[ri*3 + 1] << 16 |
									(int64_t)idx_array[ri*3 + 2] << 32;
#else
				const int64_t raw = ((int64_t)idx_high[ri] << 16) | idx_low[ri];
#endif

				const int64_t c1 = raw >> LOG_PACKING_EDGE_LISTS;
				const int64_t word_idx = c1 / NUMBER_PACKING_EDGE_LISTS;

				__builtin_prefetch(&shared_visited[word_idx], 1);

				tmp_edges[i].v0 = cur_v0_local;
				tmp_edges[i].word_idx = word_idx;
				tmp_edges[i].c1 = c1;
			}
			if(end < PREFETCH_DIST) {
				// new row
				if(new_row_idx == num_new_rows) {
					new_row_idx = 0;
					if(num_new_rows == 0 || (num_new_rows = bfs_forward_fill_new_rows(p)) == 0) {
						// There are no more rows. Shrink edges.
						active_edges = end;
						break;
					}
				}
/*
				// TODO: prefetch bv_offset, performance evaluation
				// TODO: prefetch index_array, performance evaluation
				int64_t bv_offset = p->row_array[new_row_idx + PREFETCH_DIST].bv_offset;
				__builtin_prefetch(&p->bv_row_starts[bv_offset], 0);
				int64_t row_start = p->bv_row_starts[p->row_array[new_row_idx + PREFETCH_DIST/2].bv_offset];
				__builtin_prefetch(&p->idx_high[row_start], 0);
				__builtin_prefetch(&p->idx_low[row_start], 0);
*/
				bv_offset = p->row_array[new_row_idx].bv_offset;
				ri = p->bv_row_starts[bv_offset];
				row_end = p->bv_row_starts[bv_offset + 1];
				cur_v0_local = p->row_array[new_row_idx].v;
				++new_row_idx;
				continue;
			}
			break;
		} while(1);

		// prefetch row_array, TODO: performance evaluation
	//	__builtin_prefetch(&p->row_array[new_row_idx], 0);

		// TODO: optimize using conditional move
#ifdef __FUJITSU
//#pragma loop unroll 8
#endif
		for(i = 0; i < active_edges; ++i) {
			int64_t word_idx = tmp_edges[i].word_idx;
			int bit_idx = tmp_edges[i].c1 % NUMBER_PACKING_EDGE_LISTS;
			BitmapType mask = (BitmapType)1 << bit_idx;

			if((shared_visited[word_idx] & mask) ||
			 (atomic_or_64_(&shared_visited[word_idx], mask) & mask))
			{
				continue;
			}

			int64_t c1 = tmp_edges[i].c1;
			int64_t v1_local = c1 & local_verts_mask;
			int64_t dest_c = c1 >> log_local_verts;

			FoldPacket* packet = &packet_array[dest_c];
			SET_FOLDEDGE(packet->edges[packet->num_edges], tmp_edges[i].v0, v1_local);
//			packet->v0_list[packet->num_edges] = tmp_edges[i].v0;
//			packet->v1_list[packet->num_edges] = v1_local;

			if(++packet->num_edges == PACKET_LENGTH) {
				submit_packet(context, dest_c);
				packet->num_edges = 0;
			}
		}
#if VERVOSE_MODE
		num_edge_relax += active_edges;
#endif
	} while(active_edges == PREFETCH_DIST);
#if VERVOSE_MODE
	p->num_edge_relax += num_edge_relax;
#endif
}

int bfs_backward_fill_new_rows(BfsExtractContext* p)
{
#if PROFILING_MODE
	int64_t tk_vscan = get_time_in_microsecond();
#endif
	BitmapType* visited_bitmap = p->visited_bitmap;
	BitmapType* visited_summary = p->visited_summary;
	BitmapType* bv_bitmap = p->bv_bitmap;
	int64_t* bv_offsets = p->bv_offsets;
	uint8_t* bit_scan_table = p->bit_scan_table;
	const int64_t local_verts_mask = p->local_verts_mask;

	RowPairData* row_array = p->row_array;
	int offset = 0;

	BitmapType cur_summary = p->cur_summary;
	int64_t summary_idx = p->summary_idx;
	int summary_bit_idx = p->summary_bit_idx;

	for( ; summary_bit_idx < 64; ++summary_bit_idx) {
		if((cur_summary & ((BitmapType)1 << summary_bit_idx)) == 0) {
			int unvisited_count = p->unvisited_count;
			int visited_offset = p->visited_offset;
			int64_t visited_base = ((int64_t)summary_idx*64 + summary_bit_idx)*NUMBER_CQ_SUMMARIZING;

			// prefetch
		//	__builtin_prefetch(&visited_bitmap[visited_base + NUMBER_CQ_SUMMARIZING], 0);
		//	__builtin_prefetch(&bv_bitmap[visited_base + NUMBER_CQ_SUMMARIZING], 0);
		//	__builtin_prefetch(&bv_offsets[visited_base + NUMBER_CQ_SUMMARIZING], 0);

			// TODO: prefetch row_array

			for(; visited_offset < NUMBER_CQ_SUMMARIZING; ++visited_offset) {
				int64_t e0 = visited_base + visited_offset;
				// Ignore vertices whose edges we do not have.
				BitmapType bitmap_k = ~(visited_bitmap[e0]) & bv_bitmap[e0];
				if(bitmap_k == 0) continue;
				p->num_active_vertices += popcount_64_(bitmap_k);

				++unvisited_count;
				const int64_t v1_high = (e0 * NUMBER_PACKING_EDGE_LISTS) & local_verts_mask;
				const BitmapType bv_bits = bv_bitmap[e0];
				const int64_t bv_offset_base = bv_offsets[e0];

				int row_lowbits = -1;
				while(bitmap_k) {
					int scan_bits = bitmap_k & BIT_SCAN_TABLE_MASK;
					int lsb = bit_scan_table[scan_bits];
					bitmap_k >>= lsb;
					row_lowbits += lsb;
					if(scan_bits) {
						const BitmapType bv = (bv_bits & (((BitmapType)1 << row_lowbits) - 1));
						const int64_t bv_offset = bv_offset_base + popcount_64_(bv);
						// FCC will insert software prefetch here.
						row_array[offset].bv_offset = bv_offset;
						row_array[offset].v = v1_high | row_lowbits;
						++offset;
					}
				}

				// We do not need to update visited_bitmap because it is updated in expand phase.
				// visited_bitmap[e0] |= bitmap_k;

				if(offset > BNL_ARRAY_FILL_LENGTH) {
					p->summary_bit_idx = summary_bit_idx;
					p->unvisited_count = unvisited_count;
					p->visited_offset = visited_offset;
#if PROFILING_MODE
					p->time_vertex_scan += get_time_in_microsecond() - tk_vscan;;
#endif
					return offset;
				}
			} // for(int k = 0; k < NUMBER_CQ_SUMMARIZING; ++k) {
			if(unvisited_count == 0) {
				// update summary
				visited_summary[summary_idx] |= ((BitmapType)1 << summary_bit_idx);
			} // if(unvisited_count == 0) {
			p->visited_offset = 0;
			p->unvisited_count = 0;
		} // if(summary_i & (BitmapType(1) << ii)) {
	} // for(int ii = 0; ii < (int)sizeof(visited_summary[0])*8; ++ii) {

	p->summary_bit_idx = summary_bit_idx;
#if PROFILING_MODE
	p->time_vertex_scan += get_time_in_microsecond() - tk_vscan;;
#endif
	return offset;
}

void bfs_backward_batch_prefetch(
		BfsExtractContext* p,
		int base,
		int num_rows)
{
	enum {
		WORDS_CACHE_64 = CACHE_LINE / 8,
		WORDS_CACHE_32 = CACHE_LINE / 4,
	};
	int dist_bv_offset = base + PREFETCH_DIST;
	int dist_row_start = base + PREFETCH_DIST/2;

	int i;
	// prefetch bv_offset
	int end = (num_rows < dist_bv_offset) ? num_rows : dist_bv_offset;
	for(i = 0; i < end; ++i) {
		const int64_t bv_offset = p->row_array[i].bv_offset;
		__builtin_prefetch(&p->bv_row_starts[bv_offset], 0);
	}
	// prefetch index_array
	end = (num_rows < dist_row_start) ? num_rows : dist_row_start;
	for(i = 0; i < end; ++i) {
		const int64_t row_start = p->bv_row_starts[p->row_array[i].bv_offset];
#if EDGES_IN_RAIL
		__builtin_prefetch(&p->idx_array[row_start*3], 0);
#else
		__builtin_prefetch(&p->idx_high[row_start], 0);
		__builtin_prefetch(&p->idx_low[row_start], 0);
#endif
	}
}

inline void bfs_backward_init_new_row(
		BackwardRowState* state,
		BfsExtractContext* p,
		int idx)
{
	const int64_t bv_offset = p->row_array[idx].bv_offset;
	state->v1_local = p->row_array[idx].v;
	state->ri = p->bv_row_starts[bv_offset]; // read access
	state->row_end = p->bv_row_starts[bv_offset + 1];
}

void bfs_backward_extract_edge(FoldPacket* packet, BfsExtractContext* p)
{
	BitmapType* cq_bitmap = p->cq_bitmap;
#if EDGES_IN_RAIL
	uint16_t *idx_array = p->idx_array;
#else
	int32_t *idx_high = p->idx_high;
	uint16_t *idx_low = p->idx_low;
#endif
	const int log_r = p->log_r;
	const int64_t swizzed_r = p->swizzed_r;
	const int64_t local_verts_mask = p->local_verts_mask;
	void (*submit_packet)(void*) = p->submit_packetb;
	void* context = p->context;
	int64_t *bv_row_starts = p->bv_row_starts;
	RowPairData* row_array = p->row_array;
#if VERVOSE_MODE
	int64_t num_edge_relax = 0;
#endif
	int pk_num_edges = packet->num_edges;

	enum {
		WORDS_CACHE_64 = CACHE_LINE / 8,
		WORDS_CACHE_32 = CACHE_LINE / 4,
	};

	BackwardRowState row_state[PREFETCH_DIST];

	// initialize filling new rows state
	p->summary_bit_idx = 0;
	p->unvisited_count = 0;
	p->visited_offset = 0;

	// fill new rows
	int num_new_rows = bfs_backward_fill_new_rows(p);

	// prefetch head data
	bfs_backward_batch_prefetch(p, PREFETCH_DIST, num_new_rows);

	// fill row_state
	int i, active_rows = (num_new_rows < PREFETCH_DIST) ? num_new_rows : PREFETCH_DIST;
	for(i = 0; i < active_rows; ++i) {
	//	bfs_backward_init_new_row(&row_state[i], p, i);
		const int64_t bv_offset = row_array[i].bv_offset;
		row_state[i].v1_local = row_array[i].v;
		row_state[i].ri = bv_row_starts[bv_offset]; // read access
		row_state[i].row_end = bv_row_starts[bv_offset + 1];
	}
	int new_row_idx = active_rows;

	// Fast mode
	while(new_row_idx + PREFETCH_DIST <= num_new_rows) {

#if 1
		int64_t ri, next_ri = row_state[0].ri;
		// progress edge and prefetch bitmap
		for(i = 0; i < PREFETCH_DIST-1; ++i) {
		    ri = next_ri;
		    next_ri = row_state[i+1].ri;

			// read access
			const int64_t raw = (int64_t)idx_array[ri*3 + 0] |
								(int64_t)idx_array[ri*3 + 1] << 16 |
								(int64_t)idx_array[ri*3 + 2] << 32;

			const int64_t c0 = raw >> LOG_PACKING_EDGE_LISTS;
			const int64_t word_idx = c0 / NUMBER_PACKING_EDGE_LISTS;

			__builtin_prefetch(&cq_bitmap[word_idx],
					0 /* read access */, 2 /* medium-high locality */);

			row_state[i].c0 = c0;
			row_state[i].word_idx = word_idx;
		}

	    ri = next_ri;
		// read access
		const int64_t raw = (int64_t)idx_array[ri*3 + 0] |
							(int64_t)idx_array[ri*3 + 1] << 16 |
							(int64_t)idx_array[ri*3 + 2] << 32;

		const int64_t c0 = raw >> LOG_PACKING_EDGE_LISTS;
		const int64_t word_idx = c0 / NUMBER_PACKING_EDGE_LISTS;

		__builtin_prefetch(&cq_bitmap[word_idx],
				0 /* read access */, 2 /* medium-high locality */);

		row_state[i].c0 = c0;
		row_state[i].word_idx = word_idx;
#else
		// progress edge and prefetch bitmap
		for(i = 0; i < PREFETCH_DIST; ++i) {
			const int64_t ri = row_state[i].ri;

			// read access
#if EDGES_IN_RAIL
			const int64_t raw = (int64_t)idx_array[ri*3 + 0] |
								(int64_t)idx_array[ri*3 + 1] << 16 |
								(int64_t)idx_array[ri*3 + 2] << 32;
#else
			const int64_t raw = ((int64_t)idx_high[ri] << 16) | idx_low[ri];
#endif

			const int64_t c0 = raw >> LOG_PACKING_EDGE_LISTS;
			const int64_t word_idx = c0 / NUMBER_PACKING_EDGE_LISTS;

			__builtin_prefetch(&cq_bitmap[word_idx],
					0 /* read access */, 2 /* medium-high locality */);

			row_state[i].c0 = c0;
			row_state[i].word_idx = word_idx;
		}
#endif
#if DETAILED_PROF_MODE
		int64_t core_start = get_time_in_microsecond();
#endif

		// prefetch packet, sequential write (a little effect)
		__builtin_prefetch(&packet->edges[pk_num_edges + WORDS_CACHE_64*2], 1);

		__builtin_prefetch(&row_array[new_row_idx + PREFETCH_DIST + WORDS_CACHE_64], 0);

		// read bitmap and process edges
		for(i = 0; i < PREFETCH_DIST; ++i) {
			const int64_t c0 = row_state[i].c0;
			const int64_t word_idx = row_state[i].word_idx;
			const int bit_idx = c0 % NUMBER_PACKING_EDGE_LISTS;

			// read access
			if((cq_bitmap[word_idx] & ((BitmapType)1 << bit_idx)) == 0) {
				if(++(row_state[i].ri) < row_state[i].row_end)
					continue;
			}
			else {
				// equivalent to : (c0 & ~local_verts_mask) * mpi.size_2dr
				const int64_t v0_c = (c0 & ~local_verts_mask) << log_r;
				const int64_t v0_swizzled = v0_c | swizzed_r | (c0 & local_verts_mask);
				// (e0 * NUMBER_PACKING_EDGE_LISTS) | row_lowbits : local + R

				SET_FOLDEDGE(packet->edges[pk_num_edges], v0_swizzled, row_state[i].v1_local);
		//		packet->v0_list[pk_num_edges] = v0_swizzled; // sorted
		//		packet->v1_list[pk_num_edges] = row_state[i].v1_local; // original
				++pk_num_edges;
			}

			int64_t bv_offset = row_array[new_row_idx + PREFETCH_DIST].bv_offset;
			__builtin_prefetch(&bv_row_starts[bv_offset], 0);
			int64_t row_start = bv_row_starts[row_array[new_row_idx + PREFETCH_DIST/2].bv_offset];
#if EDGES_IN_RAIL
			__builtin_prefetch(&idx_array[row_start*3], 0);
#else
			__builtin_prefetch(&idx_high[row_start], 0);
			__builtin_prefetch(&idx_low[row_start], 0);
#endif

			// new row
		//	bfs_backward_init_new_row(&row_state[i], p, new_row_idx++);

			bv_offset = row_array[new_row_idx].bv_offset;
			row_state[i].v1_local = row_array[new_row_idx].v;
			row_state[i].ri = bv_row_starts[bv_offset]; // read access
			row_state[i].row_end = bv_row_starts[bv_offset + 1];
			++new_row_idx;
		}
#if DETAILED_PROF_MODE
		p->time_core_proc += get_time_in_microsecond() - core_start;
#endif
		if(pk_num_edges >= PACKET_LENGTH - PREFETCH_DIST) {
			packet->num_edges = pk_num_edges;
			submit_packet(context);
			pk_num_edges = 0;
		}

#if VERVOSE_MODE
		num_edge_relax += PREFETCH_DIST;
#endif
	}

	while(active_rows > 0) {

		// prefetch packet, TODO: performance evaluation
	//	__builtin_prefetch(&packet->v0_list[pk_num_edges], 1);
	//	__builtin_prefetch(&packet->v1_list[pk_num_edges], 1);

		// progress edge and prefetch bitmap
		for(i = 0; i < active_rows; ) {
			const int64_t ri = row_state[i].ri;
			const int64_t row_end = row_state[i].row_end;
			if(ri == row_end) {
				// new row
				if(new_row_idx == num_new_rows) {
					new_row_idx = 0;
					if(num_new_rows == 0 || (num_new_rows = bfs_backward_fill_new_rows(p)) == 0) {
						// There are no more rows. Shrink row states.
						row_state[i] = row_state[--active_rows];
						continue;
					}
					// prefetch head data
					bfs_backward_batch_prefetch(p, 0, num_new_rows);
				}
				// prefetch bv_offset
				if(new_row_idx + PREFETCH_DIST < num_new_rows) {
					const int64_t bv_offset = row_array[new_row_idx + PREFETCH_DIST].bv_offset;
					__builtin_prefetch(&bv_row_starts[bv_offset], 0);
				}
				// prefetch index_array
				if(new_row_idx + PREFETCH_DIST/2 < num_new_rows) {
					const int64_t row_start =
							bv_row_starts[row_array[new_row_idx + PREFETCH_DIST/2].bv_offset];
#if EDGES_IN_RAIL
					__builtin_prefetch(&idx_array[row_start*3], 0);
#else
					__builtin_prefetch(&idx_high[row_start], 0);
					__builtin_prefetch(&idx_low[row_start], 0);
#endif
				}
				const int64_t bv_offset = row_array[new_row_idx].bv_offset;
				row_state[i].v1_local = row_array[new_row_idx].v;
				row_state[i].ri = bv_row_starts[bv_offset]; // read access
				row_state[i].row_end = bv_row_starts[bv_offset + 1];
				++new_row_idx;
				continue;
			}

			// TODO: split loop for optimization

			// read access
#if EDGES_IN_RAIL
			const int64_t raw = (int64_t)idx_array[ri*3 + 0] |
								(int64_t)idx_array[ri*3 + 1] << 16 |
								(int64_t)idx_array[ri*3 + 2] << 32;
#else
			const int64_t raw = ((int64_t)idx_high[ri] << 16) | idx_low[ri];
#endif

			const int64_t c0 = raw >> LOG_PACKING_EDGE_LISTS;
			const int64_t word_idx = c0 / NUMBER_PACKING_EDGE_LISTS;

			__builtin_prefetch(&cq_bitmap[word_idx],
					0 /* read access */, 2 /* medium-high locality */);

			row_state[i].c0 = c0;
			row_state[i].word_idx = word_idx;
			++i;
		}

#if DETAILED_PROF_MODE
		int64_t core_start = get_time_in_microsecond();
#endif
		// TODO: optimize using conditional move
		// read bitmap and process edges
		for(i = 0; i < active_rows; ++i) {
			const int64_t c0 = row_state[i].c0;
			const int64_t word_idx = row_state[i].word_idx;
			const int bit_idx = c0 % NUMBER_PACKING_EDGE_LISTS;

			// read access
			if((cq_bitmap[word_idx] & ((BitmapType)1 << bit_idx)) == 0) {
				++(row_state[i].ri);
				continue;
			}

			// equivalent to : (c0 & ~local_verts_mask) * mpi.size_2dr
			const int64_t v0_c = (c0 & ~local_verts_mask) << log_r;
			const int64_t v0_swizzled = v0_c | swizzed_r | (c0 & local_verts_mask);
			// (e0 * NUMBER_PACKING_EDGE_LISTS) | row_lowbits : local + R

			SET_FOLDEDGE(packet->edges[pk_num_edges], v0_swizzled, row_state[i].v1_local);
		//	packet->v0_list[pk_num_edges] = v0_swizzled; // sorted
		//	packet->v1_list[pk_num_edges] = row_state[i].v1_local; // original
			++pk_num_edges;

			row_state[i].ri = row_state[i].row_end;
		}
#if DETAILED_PROF_MODE
		p->time_core_proc += get_time_in_microsecond() - core_start;
#endif

		if(pk_num_edges >= PACKET_LENGTH - PREFETCH_DIST) {
			packet->num_edges = pk_num_edges;
			submit_packet(context);
			pk_num_edges = 0;
		}

#if VERVOSE_MODE
		num_edge_relax += active_rows;
#endif
	}

	packet->num_edges = pk_num_edges;
#if VERVOSE_MODE
	p->num_edge_relax += num_edge_relax;
#endif
}

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
		)
{
#define UNSWIZZLE_VERTEX(c) (((c) >> log_local_verts) | (((c) & local_verts_mask) << log_size))
	int64_t num_nq_vertices = 0;
	int c;
	struct { int64_t v1_local, pred_v; } write_buffer[PREFETCH_DIST];

#if 0
	for(c = 0; c < num_edges; ++c) {
		int64_t v0_swizzled = v0_list[c]; // long sequential access
		const uint32_t v1_local = v1_list[c]; // long sequential access
		const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t mask = (uint64_t)1 << bit_idx;
		if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
			if((atomic_or_64_(&visited[word_idx], mask) & mask) == 0) {
				const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
				const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
				assert (pred[v1_local] == -1);
			//	atomic_or_64_(&nq_bitmap[word_idx], mask);
			//	pred[v1_local] = pred_v;
				__builtin_prefetch(&nq_bitmap[word_idx], 1);
				__builtin_prefetch(&pred[v1_local], 1);
				write_buffer[num_nq_vertices].v1_local = v1_local;
				write_buffer[num_nq_vertices].pred_v = pred_v;
				if(++num_nq_vertices == PREFETCH_DIST) break;
			}
		}
	}
	for( ; c < num_edges - PREFETCH_DIST; ++c) {
		{
			const uint32_t v1_local = v1_list[c + PREFETCH_DIST];
			const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
			__builtin_prefetch(&visited[word_idx], 0);
		}
		int64_t v0_swizzled = v0_list[c]; // long sequential access
		const uint32_t v1_local = v1_list[c]; // long sequential access
		const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t mask = (uint64_t)1 << bit_idx;
		if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
			if((atomic_or_64_(&visited[word_idx], mask) & mask) == 0) {
				const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
				const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
				assert (pred[v1_local] == -1);
				atomic_or_64_(&nq_bitmap[word_idx], mask);
			//	pred[v1_local] = pred_v;

				int write_buf_idx = num_nq_vertices % PREFETCH_DIST;
				const uint32_t o_v1_local = write_buffer[write_buf_idx].v1_local; // long sequential access
				const uint32_t o_word_idx = o_v1_local / NUMBER_PACKING_EDGE_LISTS;
				const int o_bit_idx = o_v1_local % NUMBER_PACKING_EDGE_LISTS;
				const uint64_t o_mask = (uint64_t)1 << o_bit_idx;

				atomic_or_64_(&nq_bitmap[o_word_idx], o_mask);
				pred[o_v1_local] = write_buffer[write_buf_idx].pred_v;

				__builtin_prefetch(&nq_bitmap[word_idx], 1);
				__builtin_prefetch(&pred[v1_local], 1);

				write_buffer[write_buf_idx].v1_local = v1_local;
				write_buffer[write_buf_idx].pred_v = pred_v;
				++num_nq_vertices;
			}
		}
	}
	for( ; c < num_edges; ++c) {
		int64_t v0_swizzled = v0_list[c]; // long sequential access
		const uint32_t v1_local = v1_list[c]; // long sequential access
		const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t mask = (uint64_t)1 << bit_idx;
		if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
			if((atomic_or_64_(&visited[word_idx], mask) & mask) == 0) {
				const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
				const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
				assert (pred[v1_local] == -1);
				atomic_or_64_(&nq_bitmap[word_idx], mask);
			//	pred[v1_local] = pred_v;

				int write_buf_idx = num_nq_vertices % PREFETCH_DIST;
				const uint32_t o_v1_local = write_buffer[write_buf_idx].v1_local; // long sequential access
				const uint32_t o_word_idx = o_v1_local / NUMBER_PACKING_EDGE_LISTS;
				const int o_bit_idx = o_v1_local % NUMBER_PACKING_EDGE_LISTS;
				const uint64_t o_mask = (uint64_t)1 << o_bit_idx;

				atomic_or_64_(&nq_bitmap[o_word_idx], o_mask);
				pred[o_v1_local] = write_buffer[write_buf_idx].pred_v;

				__builtin_prefetch(&nq_bitmap[word_idx], 1);
				__builtin_prefetch(&pred[v1_local], 1);

				write_buffer[write_buf_idx].v1_local = v1_local;
				write_buffer[write_buf_idx].pred_v = pred_v;
				++num_nq_vertices;
			}
		}
	}
	int num_write_entry = (num_nq_vertices < PREFETCH_DIST) ? num_nq_vertices : PREFETCH_DIST;
	for(c = 0; c < num_write_entry; ++c) {
		const uint32_t o_v1_local = write_buffer[c].v1_local; // long sequential access
		const uint32_t o_word_idx = o_v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int o_bit_idx = o_v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t o_mask = (uint64_t)1 << o_bit_idx;

		atomic_or_64_(&nq_bitmap[o_word_idx], o_mask);
		pred[o_v1_local] = write_buffer[c].pred_v;
	}
#else
	for(c = 0; c < num_edges; ++c) {
		int64_t v0_swizzled;
		uint32_t v1_local;
		GET_FOLDEDGE(edges[c], v0_swizzled, v1_local);
		const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t mask = (uint64_t)1 << bit_idx;
		if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
			if((atomic_or_64_(&visited[word_idx], mask) & mask) == 0) {
				const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
				const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
				assert (pred[v1_local] == -1);
				atomic_or_64_(&nq_bitmap[word_idx], mask);
			//	pred[v1_local] = pred_v;
				__builtin_prefetch(&pred[v1_local], 1);
				write_buffer[num_nq_vertices].v1_local = v1_local;
				write_buffer[num_nq_vertices].pred_v = pred_v;
				if(++num_nq_vertices == PREFETCH_DIST) break;
			}
		}
	}
	for( ; c < num_edges; ++c) {
		int64_t v0_swizzled;
		uint32_t v1_local;
		GET_FOLDEDGE(edges[c], v0_swizzled, v1_local);
		const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t mask = (uint64_t)1 << bit_idx;
		if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
			if((atomic_or_64_(&visited[word_idx], mask) & mask) == 0) {
				const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
				const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
				assert (pred[v1_local] == -1);
				atomic_or_64_(&nq_bitmap[word_idx], mask);
			//	pred[v1_local] = pred_v;
				int write_buf_idx = num_nq_vertices % PREFETCH_DIST;
				pred[write_buffer[write_buf_idx].v1_local] = write_buffer[write_buf_idx].pred_v;
				__builtin_prefetch(&pred[v1_local], 1);
				write_buffer[write_buf_idx].v1_local = v1_local;
				write_buffer[write_buf_idx].pred_v = pred_v;
				++num_nq_vertices;
			}
		}
	}
	int num_write_entry = (num_nq_vertices < PREFETCH_DIST) ? num_nq_vertices : PREFETCH_DIST;
	for(c = 0; c < num_write_entry; ++c) {
		pred[write_buffer[c].v1_local] = write_buffer[c].pred_v;
	}
#endif

#undef UNSWIZZLE_VERTEX
	return num_nq_vertices;
}

int64_t bfs_receiver_processing_v1(
		const int64_t* const v0_list,
		const uint32_t* const v1_list,
		const int num_edges,
		uint64_t* const visited,
		uint64_t* const nq_bitmap,
		int64_t* const pred,
		const int cur_level,
		const int log_local_verts,
		const int log_size,
		const int64_t local_verts_mask
		)
{
#define UNSWIZZLE_VERTEX(c) (((c) >> log_local_verts) | (((c) & local_verts_mask) << log_size))
	int64_t num_nq_vertices = 0;
	int c;
	for(c = 0; c < num_edges; ++c) {
		/*
		if(c < num_edges - PREFETCH_DIST) {
			const uint32_t v1_local = v1_list[c + PREFETCH_DIST];
			const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
			__builtin_prefetch(&visited[word_idx], 0);
			__builtin_prefetch(&nq_bitmap[word_idx], 0);
		//	__builtin_prefetch(&pred[v1_local], 0);
		}
		*/

		int64_t v0_swizzled = v0_list[c]; // long sequential access
		const uint32_t v1_local = v1_list[c]; // long sequential access
		const uint32_t word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint64_t mask = (uint64_t)1 << bit_idx;

		// TODO: prefetch visited and nq_bitmap

		if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
			if((atomic_or_64_(&visited[word_idx], mask) & mask) == 0) {
				const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
				const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
				assert (pred[v1_local] == -1);
				atomic_or_64_(&nq_bitmap[word_idx], mask);
				pred[v1_local] = pred_v;
				++num_nq_vertices;
			}
		}
	}
#undef UNSWIZZLE_VERTEX
	return num_nq_vertices;
}



