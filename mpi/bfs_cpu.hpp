/*
 * bfs_cpu.hpp
 *
 *  Created on: Mar 21, 2012
 *      Author: koji
 */

#ifndef BFS_CPU_HPP_
#define BFS_CPU_HPP_

#include "bfs.hpp"

struct BfsOnCPU_Params {
	typedef uint64_t BitmapType;
	enum {
		LOG_PACKING_EDGE_LISTS = 6, // 2^6 = 64
		LOG_CQ_SUMMARIZING = 4, // 2^4 = 16 -> sizeof(int64_t)*32 = 128bytes
	};
};

template <typename IndexArray, typename LocalVertsIndex>
class BfsOnCPU
	: public BfsBase<IndexArray, LocalVertsIndex, BfsOnCPU_Params>
{
public:
	BfsOnCPU()
	: BfsBase<IndexArray, LocalVertsIndex, BfsOnCPU_Params>(false)
	  { }
};

#endif /* BFS_CPU_HPP_ */
