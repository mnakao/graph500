#ifndef BFS_HPP_
#define BFS_HPP_
#include <pthread.h>
#include <deque>
#include "utils.hpp"
#include "abstract_comm.hpp"

class BfsBase
{
public:
	enum {
		ENABLE_WRITING_DEPTH = 1,
		BUCKET_UNIT_SIZE = 1024,
		NBPE = PRM::NBPE,
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK,
		BFELL_SORT = PRM::BFELL_SORT,
		LOG_BFELL_SORT = PRM::LOG_BFELL_SORT,
		BFELL_SORT_MASK = PRM::BFELL_SORT_MASK,
		BFELL_SORT_IN_BMP = BFELL_SORT / NBPE,
		BU_SUBSTEP = PRM::NUM_BOTTOM_UP_STREAMS,
	};

	BfsBase(){}
	virtual ~BfsBase(){}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		int log_local_verts_unit = get_msb_index(std::max<int>(BFELL_SORT, NBPE) * 8);
		detail::GraphConstructor2DCSR<EdgeList> constructor;
		Graph2DCSR graph_;
		constructor.construct(edge_list, log_local_verts_unit, graph_);
	}
};

#endif /* BFS_HPP_ */

