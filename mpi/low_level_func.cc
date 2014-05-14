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
#include <omp.h>

#include <algorithm>

#include "low_level_func.h"

#if LOW_LEVEL_FUNCTION

void backward_isolated_edge(
	int half_bitmap_width,
	int phase_bmp_off,
	BitmapType* __restrict__ phase_bitmap,
	const BitmapType* __restrict__ row_bitmap,
	const BitmapType* __restrict__ shared_visited,
	const TwodVertex* __restrict__ row_sums,
	const TwodVertex* __restrict__ isolated_edges,
	const int64_t* __restrict__ row_starts,
	const TwodVertex* __restrict__ edge_array,
	LocalPacket* buffer
) {
	int tid = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	int width_per_thread = (half_bitmap_width + num_threads - 1) / num_threads;
	int off_start = std::min(half_bitmap_width, width_per_thread * tid);
	int off_end = std::min(half_bitmap_width, off_start + width_per_thread);
	int num_send = 0;

#if 1 // consolidated
	for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
		BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
		BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
		TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
		BitmapType bit_flags = (~visited_i) & row_bmp_i;
		while(bit_flags != BitmapType(0)) {
			BitmapType vis_bit = bit_flags & (-bit_flags);
			BitmapType mask = vis_bit - 1;
			bit_flags &= ~vis_bit;
			int idx = __builtin_popcountl(mask);
			TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
			// short cut
			TwodVertex src = isolated_edges[non_zero_idx];
			if(shared_visited[src >> PRM::LOG_NBPE] & (BitmapType(1) << (src & PRM::NBPE_MASK))) {
				// add to next queue
				visited_i |= vis_bit;
				buffer->data.b[num_send+0] = src;
				buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
				num_send += 2;
				// end this row
				continue;
			}
			int64_t e_start = row_starts[non_zero_idx];
			int64_t e_end = row_starts[non_zero_idx+1];
			for(int64_t e = e_start; e < e_end; ++e) {
				TwodVertex src = edge_array[e];
				if(shared_visited[src >> PRM::LOG_NBPE] & (BitmapType(1) << (src & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send+0] = src;
					buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
					num_send += 2;
					// end this row
					break;
				}
			}
		} // while(bit_flags != BitmapType(0)) {
		// write back
		*(phase_bitmap + blk_bmp_off) = visited_i;
	} // #pragma omp for

#else // separated
	for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
		BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
		BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
		TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
		BitmapType bit_flags = (~visited_i) & row_bmp_i;
		while(bit_flags != BitmapType(0)) {
			BitmapType vis_bit = bit_flags & (-bit_flags);
			BitmapType mask = vis_bit - 1;
			bit_flags &= ~vis_bit;
			int idx = __builtin_popcountl(mask);
			TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
			// short cut
			TwodVertex src = isolated_edges[non_zero_idx];
			if(shared_visited[src >> PRM::LOG_NBPE] & (BitmapType(1) << (src & PRM::NBPE_MASK))) {
				// add to next queue
				visited_i |= vis_bit;
				buffer->data.b[num_send+0] = src;
				buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
				num_send += 2;
			}
		} // while(bit_flags != BitmapType(0)) {
		// write back
		*(phase_bitmap + blk_bmp_off) = visited_i;
	} // #pragma omp for

	for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
		BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
		BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
		TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
		BitmapType bit_flags = (~visited_i) & row_bmp_i;
		while(bit_flags != BitmapType(0)) {
			BitmapType vis_bit = bit_flags & (-bit_flags);
			BitmapType mask = vis_bit - 1;
			bit_flags &= ~vis_bit;
			int idx = __builtin_popcountl(mask);
			TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
			int64_t e_start = row_starts[non_zero_idx];
			int64_t e_end = row_starts[non_zero_idx+1];
			for(int64_t e = e_start; e < e_end; ++e) {
				TwodVertex src = edge_array[e];
				if(shared_visited[src >> PRM::LOG_NBPE] & (BitmapType(1) << (src & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send+0] = src;
					buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
					num_send += 2;
					// end this row
					break;
				}
			}
		} // while(bit_flags != BitmapType(0)) {
		// write back
		*(phase_bitmap + blk_bmp_off) = visited_i;
	} // #pragma omp for
#endif

	buffer->length = num_send;
}


#endif // #if LOW_LEVEL_FUNCTION

