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
	enum {
		LOG_PACKING_EDGE_LISTS = 6, // 2^6 = 64
		LOG_CQ_SUMMARIZING = 4, // 2^4 = 16 -> sizeof(int64_t)*32 = 128bytes
	};
};

class BfsOnCPU
	: public BfsBase<BfsOnCPU_Params>
{
public:
	BfsOnCPU(double demon_to_bottom_up__, double demon_to_top_down__)
	: BfsBase<BfsOnCPU_Params>(demon_to_bottom_up__, demon_to_top_down__, false)
	  { }
};

#endif /* BFS_CPU_HPP_ */
