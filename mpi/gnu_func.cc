/*
 * gnu_func.c
 *
 *  Created on: Apr 6, 2012
 *      Author: koji
 */

#include <stdint.h>

int32_t atomicAdd_(volatile int32_t* ptr, int32_t n) {
	return __sync_fetch_and_add(ptr, n);
}
int64_t atomicAdd_(volatile int64_t* ptr, int64_t n) {
	return __sync_fetch_and_add(ptr, n);
}

uint32_t atomicOr_(volatile uint32_t* ptr, uint32_t n) {
	return __sync_fetch_and_or(ptr, n);
}
uint64_t atomicOr_(volatile uint64_t* ptr, uint64_t n) {
	return __sync_fetch_and_or(ptr, n);
}

