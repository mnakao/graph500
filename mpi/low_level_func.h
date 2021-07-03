/*
 * low_level_func.h
 *
 *  Created on: 2012/10/17
 *      Author: ueno
 */

#ifndef LOW_LEVEL_FUNC_H_
#define LOW_LEVEL_FUNC_H_

#include "parameters.h"

struct LocalPacket {
	enum {
		TOP_DOWN_LENGTH = PRM::PACKET_LENGTH/sizeof(uint32_t),
		BOTTOM_UP_LENGTH = PRM::PACKET_LENGTH/sizeof(int64_t)
	};
	int length;
	int64_t src;
	union {
		uint32_t t[TOP_DOWN_LENGTH];
		int64_t b[BOTTOM_UP_LENGTH];
	} data;
};

#endif /* LOW_LEVEL_FUNC_H_ */
