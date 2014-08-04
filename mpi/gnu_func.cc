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

bool compareAndSwap_(volatile int64_t* ptr, int64_t old_value, int64_t new_value) {
	return __sync_bool_compare_and_swap(ptr, old_value, new_value);
}
bool compareAndSwap_(volatile int32_t* ptr, int32_t old_value, int32_t new_value) {
	return __sync_bool_compare_and_swap(ptr, old_value, new_value);
}

