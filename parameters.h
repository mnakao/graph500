#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#define CACHE_LINE 256
#define PAGE_SIZE 65536
#define IMD_OUT stderr

typedef uint64_t BitmapType;
typedef uint64_t TwodVertex;
typedef uint32_t LocalVertex;

namespace PRM { //
enum {
	LOG_NBPE = 6,
	LOG_BFELL_SORT = 8,
	NBPE = 1 << LOG_NBPE, // <= sizeof(BitmapType)*8
	NBPE_MASK = NBPE - 1,
	BFELL_SORT = 1 << LOG_BFELL_SORT,
	USERSEED1 = 2,
	USERSEED2 = 3,
};
}

#endif /* PARAMETERS_H_ */
