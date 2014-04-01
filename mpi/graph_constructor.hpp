/*
 * graph_constructor.hpp
 *
 *  Created on: Dec 14, 2011
 *      Author: koji
 */

#ifndef GRAPH_CONSTRUCTOR_HPP_
#define GRAPH_CONSTRUCTOR_HPP_

#include "parameters.h"

//-------------------------------------------------------------//
// 2D partitioning
//-------------------------------------------------------------//

class Graph2DCSR
{
	enum {
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK
	};
public:
	Graph2DCSR()
	: row_bitmap_(NULL)
	, row_sums_(NULL)
#if BFELL
	, blk_off(NULL)
	, sorted_idx_(NULL)
	, col_len_(NULL)
#else
	, row_starts_(NULL)
#endif
	, log_actual_global_verts_(0)
	, log_global_verts_(0)
	, log_max_weight_(0)
	, max_weight_(0)
	, num_global_edges_(0)
	, num_global_verts_(0)
	{ }
	~Graph2DCSR()
	{
		clean();
	}

	void clean()
	{
		free(row_bitmap_); row_bitmap_ = NULL;
		free(row_sums_); row_sums_ = NULL;
		free(edge_array_); edge_array_ = NULL;
#if BFELL
		free(sorted_idx_); sorted_idx_ = NULL;
		free(col_len_); col_len_ = NULL;
		free(blk_off); blk_off = NULL;
#else
		free(row_starts_); row_starts_ = NULL;
#endif
	}

	int log_actual_global_verts() const { return log_actual_global_verts_; }
	int log_actual_local_verts() const { return log_actual_global_verts_ - get_msb_index(mpi.size_2d); }
	int log_global_verts() const { return log_global_verts_; }
	int log_local_bitmap() const { return lgl_ - LOG_NBPE; }
	int log_local_verts() const { return lgl_; }
	int log_local_src() const { return lgl_ + lgc_; }
	int log_local_tgt() const { return lgl_ + lgr_; }

	// Reference Functions
	int get_vertex_rank(int64_t v)
	{
		const int64_t mask = mpi.size_2d - 1;
		return v & mask;
	}

	int get_vertex_rank_r(int64_t v)
	{
		const int64_t rmask = mpi.size_2dr - 1;
		return v & rmask;
	}

	int get_vertex_rank_c(int64_t v)
	{
		const int64_t cmask = mpi.size_2dc - 1;
		const int log_size_r = get_msb_index(mpi.size_2dr);
		return (v >> log_size_r) & cmask;
	}

	int64_t swizzle_vertex(int64_t v) {
		return ((v & (mpi.size_2d - 1)) << log_local_verts()) | (v >> get_msb_index(mpi.size_2d));
	}

	int64_t unswizzle_vertex(int64_t v) {
		const int64_t local_verts_mask = ((int64_t)1 << log_local_verts()) - 1;
		return ((v & local_verts_mask) << get_msb_index(mpi.size_2d)) |
				(v >> log_local_verts());
	}

	// vertex id converter
	int64_t VtoD(int64_t id) {
		int lgl = lgl_;
		int lgsize = lgr_ + lgc_;
		int64_t rmask = (int64_t(1) << lgr_) - 1;
		return ((id & rmask) << lgl) | (id >> lgsize);
	}
	int64_t VtoS(int64_t id) {
		int lgr = lgr_;
		int lgl = lgl_;
		int lgsize = lgr + lgc_;
		int64_t cmask = ((int64_t(1) << lgsize) - 1) - ((int64_t(1) << lgr) - 1);
		return (((id & cmask) >> lgr) << lgl) | (id >> lgsize);
	}
	int64_t DtoV(int64_t id, int c) {
		int lgl = lgl_;
		int64_t cshifted = int64_t(c) << lgr_;
		int lgsize = lgr_ + lgc_;
		int64_t lmask = ((int64_t(1) << lgl) - 1L);
		return ((id & lmask) << lgsize) | cshifted | (id >> lgl);
	}
	int64_t StoV(int64_t id, int r) {
		int lgr = lgr_;
		int lgl = lgl_;
		int lgsize = lgr + lgc_;
		int64_t lmask = ((int64_t(1) << lgl) - 1);
		return ((id & lmask) << lgsize) | ((id >> lgl) << lgr) | int64_t(r);
	}
	int64_t StoD(int64_t id, int r) {
		int64_t rshifted = int64_t(r) << lgl_;
		int64_t lmask = ((1L << lgl_) - 1L);
		return (id & lmask) | rshifted;
	}
	int64_t DtoS(int64_t id, int c) {
		int64_t cshiftedto = int64_t(c) << lgl_;
		int64_t lmask = ((int64_t(1) << lgl_) - 1);
		return (id & lmask) | cshiftedto;
	}
/*
	TwodVertex get_edge_list_index(int64_t v0)
	{
		const int64_t rmask = mpi.size_2dr - 1;
		const int log_local_verts_minus_packing_edge_lists =
				log_local_verts() - log_packing_edge_lists();
		const int log_size_plus_packing_edge_lists =
				get_msb_index(mpi.size_2d) + log_packing_edge_lists();
		return ((v0 & rmask) << log_local_verts_minus_packing_edge_lists) |
				(v0 >> log_size_plus_packing_edge_lists);
	}

	int64_t get_v0_from_edge(int64_t e0, int64_t e1)
	{
		const int log_size_r = get_msb_index(mpi.size_2dr);
		const int log_size = get_msb_index(mpi.size_2d);
		const int mask_packing_edge_lists = ((1 << log_packing_edge_lists()) - 1);

		const int packing_edge_lists = log_packing_edge_lists();
		const int log_local_verts_ = log_local_verts();
		const int64_t v0_high_mask = ((INT64_C(1) << (log_local_verts_ - packing_edge_lists)) - 1);

		const int rank_c = mpi.rank_2dc;

		int v0_r = e0 >> (log_local_verts_ - packing_edge_lists);
		int64_t v0_high = e0 & v0_high_mask;
		int64_t v0_middle = e1 & mask_packing_edge_lists;
		return (((v0_high << packing_edge_lists) | v0_middle) << log_size) | ((rank_c << log_size_r) | v0_r);
	}

	int64_t get_v1_from_edge(int64_t e1, bool has_weight = false)
	{
		const int log_size_r = get_msb_index(mpi.size_2dr);

		const int packing_edge_lists = log_packing_edge_lists();
		const int rank_r = mpi.rank_2dr;

		if(has_weight) {
			int64_t v1_and_weight = e1 >> packing_edge_lists;
			int64_t v1_high = v1_and_weight >> log_max_weight_;
			return (v1_high << log_size_r) | rank_r;
		}
		else {
			int64_t v1_high = e1 >> packing_edge_lists;
			return (v1_high << log_size_r) | rank_r;
		}
	}
*/
	int get_weight_from_edge(int64_t e)
	{
		return e & ((1 << log_max_weight_) - 1);
	}

	bool has_edge(int64_t v, bool has_weight = false)
	{
		if(get_vertex_rank(v) == mpi.rank_2d) {
			int64_t v_local = v / mpi.size_2d;
			int64_t word_idx = v_local >> LOG_NBPE;
			int bit_idx = v_local & NBPE_MASK;
			return has_edge_bitmap_[word_idx] & (BitmapType(1) << bit_idx);
		}
		return false;
	}
//private:
	BitmapType* row_bitmap_;
	TwodVertex* row_sums_;
	BitmapType* has_edge_bitmap_; // for every local vertices

	TwodVertex* edge_array_;
#if BFELL
	struct BlockOffset {
		int64_t length_start;
		int64_t edge_start;
	};
	BlockOffset* blk_off;
	SortIdx* sorted_idx_;
	SortIdx* col_len_;
#else
	int64_t* row_starts_;
#endif

	int log_actual_global_verts_;
	int log_global_verts_;
	int log_max_weight_;

	int max_weight_;
	int64_t num_global_edges_;
	int64_t num_global_verts_;

	// for id converter
	int lgr_;
	int lgc_;
	int lgl_;
};

namespace detail {

template <typename EdgeList>
class GraphConstructor2DCSR
{
	enum {
		LOG_EDGE_PART_SIZE = 16,
	//	LOG_EDGE_PART_SIZE = 14,
		EDGE_PART_SIZE = 1 << LOG_EDGE_PART_SIZE, // == UINT64_MAX + 1
		EDGE_PART_SIZE_MASK = EDGE_PART_SIZE - 1,

		NBPE = PRM::NBPE,
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK,

		BFELL_SORT = PRM::BFELL_SORT,
		LOG_BFELL_SORT = PRM::LOG_BFELL_SORT,

		BFELL_SORT_IN_BMP = PRM::BFELL_SORT / NBPE,
	};
public:
	typedef Graph2DCSR GraphType;
	typedef typename EdgeList::edge_type EdgeType;

	GraphConstructor2DCSR()
		: log_size_(get_msb_index(mpi.size_2d))
		, rmask_((1 << get_msb_index(mpi.size_2dr)) - 1)
		, cmask_((1 << get_msb_index(mpi.size_2d)) - 1 - rmask_)
		, src_vertexes_(NULL)
		, wide_row_starts_(NULL)
	{ }
	~GraphConstructor2DCSR()
	{
		free(src_vertexes_); src_vertexes_ = NULL;
		free(wide_row_starts_); wide_row_starts_ = NULL;
	}

	void construct(EdgeList* edge_list, int log_minimum_global_verts, bool sort_by_degree, GraphType& g)
	{
		VT_TRACER("construction");
		log_minimum_global_verts_ = log_minimum_global_verts;
		g.log_actual_global_verts_ = 0;

		do { // loop for max vertex estimation failure
			scatterAndScanEdges(edge_list, sort_by_degree, g);
		} while(g.log_actual_global_verts_ == 0);

		scatterAndStore(edge_list, sort_by_degree, g);
		sortEdges(g);
		free(row_starts_sup_); row_starts_sup_ = NULL;

		if(mpi.isMaster()) fprintf(IMD_OUT, "Wide CSR creation complete.\n");

		constructFromWideCSR(g);

		computeNumVertices(g);

		if(mpi.isMaster()) fprintf(IMD_OUT, "Graph construction complete.\n");
	}

	void copy_to_gpu(GraphType& g, bool graph_on_gpu_) {
#if CUDA_ENABLED
		// transfer data to GPU
		const int64_t num_columns = (int64_t(1) << g.log_edge_lists());
		const int64_t index_size = g.row_starts_[num_columns];
		const int64_t num_local_vertices = (int64_t(1) << g.log_local_verts());

		CudaStreamManager::begin_cuda();
		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMalloc((void**)&g.dev_row_starts_,
					sizeof(g.dev_row_starts_[0])*(num_columns+2)));
			CUDA_CHECK(cudaMalloc((void**)&g.dev_edge_array_high_,
					sizeof(g.dev_edge_array_high_[0])*index_size));
			CUDA_CHECK(cudaMalloc((void**)&g.dev_edge_array_low_,
					sizeof(g.dev_edge_array_low_[0])*index_size));
		}
		else {
			g.dev_row_starts_ = NULL;
			g.dev_edge_array_high_ = NULL;
			g.dev_edge_array_low_ = NULL;
		}
		CUDA_CHECK(cudaMalloc((void**)&g.dev_invert_vertex_mapping_,
				sizeof(g.dev_invert_vertex_mapping_[0])*num_local_vertices));

		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMemcpy(g.dev_row_starts_, g.row_starts_,
					sizeof(g.dev_row_starts_[0])*(num_columns+1), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(g.dev_edge_array_high_, g.edge_array_.get_ptr_high(),
					sizeof(g.dev_edge_array_high_[0])*index_size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(g.dev_edge_array_low_, g.edge_array_.get_ptr_low(),
					sizeof(g.dev_edge_array_low_[0])*index_size, cudaMemcpyHostToDevice));
			// add an empty column
			CUDA_CHECK(cudaMemcpy(g.dev_row_starts_ + num_columns + 1, &index_size,
					sizeof(g.dev_row_starts_[0]), cudaMemcpyHostToDevice));
		}
		CUDA_CHECK(cudaMemcpy(g.dev_invert_vertex_mapping_, g.invert_vertex_mapping_,
				sizeof(g.dev_invert_vertex_mapping_[0])*num_local_vertices, cudaMemcpyHostToDevice));
		CudaStreamManager::end_cuda();
#endif
	}

private:
	int edge_owner(int64_t v0, int64_t v1) const { return (v0 & rmask_) | (v1 & cmask_); }
	int vertex_owner(int64_t v) const { return v & (mpi.size_2d - 1); }
	int64_t vertex_local(int64_t v) { return v >> log_size_; }

	void initializeParameters(
		int log_max_vertex,
		int64_t num_global_edges,
		bool sort_by_degree,
		GraphType& g)
	{
		g.log_actual_global_verts_ = log_max_vertex;
		g.log_global_verts_ = std::max(log_minimum_global_verts_, log_max_vertex);

		g.lgl_ = g.log_global_verts_ - get_msb_index(mpi.size_2d);
		g.lgr_ = get_msb_index(mpi.size_2dr);
		g.lgc_ = get_msb_index(mpi.size_2dc);

		log_local_verts_ = g.log_local_verts();

		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);

		wide_row_starts_ = static_cast<int64_t*>
			(cache_aligned_xmalloc((num_wide_rows+1)*sizeof(wide_row_starts_[0])));
		row_starts_sup_ = static_cast<int64_t*>(
				cache_aligned_xmalloc((num_wide_rows+1)*sizeof(row_starts_sup_[0])));
		memset(wide_row_starts_, 0x00, (num_wide_rows+1)*sizeof(wide_row_starts_[0]));
	}

	void scanEdges(TwodVertex* edges, int64_t num_edges, GraphType& g) {
#define EDGE_PART_IDX(v) (((v) >> LOG_EDGE_PART_SIZE) + 1)
		int64_t i;
#pragma omp parallel for schedule(static)
		for(i = 0; i < (num_edges&(~3)); i += 4) {
#if 1
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+0])], 1);
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+1])], 1);
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+2])], 1);
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+3])], 1);
#else
#pragma omp atomic
			wide_row_starts_[EDGE_PART_IDX(edges[i+0])] += 1;
#pragma omp atomic
			wide_[EDGE_PART_IDX(edges[i+1])] += 1;
#pragma omp atomic
			wide_row_starts_[EDGE_PART_IDX(edges[i+2])] += 1;
#pragma omp atomic
			wide_row_starts_[EDGE_PART_IDX(edges[i+3])] += 1;
#endif
		}
		for(i = (num_edges&(~3)); i < num_edges; ++i) {
			wide_row_starts_[EDGE_PART_IDX(edges[i])]++;
		}
#undef EDGE_PART_IDX
	}

	class CountDegree
	{
	public:
		typedef TwodVertex send_type;

		CountDegree(GraphConstructor2DCSR* this__,
				UnweightedPackedEdge* edges, int64_t* degree_counts, int log_local_verts)
			: this_(this__)
			, edges_(edges)
			, degree_counts_(degree_counts)
			, log_local_verts_plus_size_r_(log_local_verts + get_msb_index(mpi.size_2dr))
			, local_verts_mask_((int64_t(1) << log_local_verts) - 1)
		{ }
		int target(int i) const {
			const int64_t v1_swizzled = edges_[i].v1();
			assert ((v1_swizzled >> log_local_verts_plus_size_r_) < mpi.size_2dc);
			return v1_swizzled >> log_local_verts_plus_size_r_;
		}
		TwodVertex get(int i) const {
			return edges_[i].v1() & local_verts_mask_;
		}
		void set(int i, TwodVertex v1) const {
#if 0
#pragma omp atomic
			degree_counts_[v1] += 1;
#else
			__sync_fetch_and_add(&degree_counts_[v1], 1);
#endif
		}
	private:
		GraphConstructor2DCSR* const this_;
		const UnweightedPackedEdge* const edges_;
		int64_t* degree_counts_;
		const int log_local_verts_plus_size_r_;
		const int64_t local_verts_mask_;
	};

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void scanEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict counts, uint64_t& max_vertex, int& max_weight, typename EdgeType::has_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			const int weight = edge_data[i].weight_;
			if (v0 == v1) continue;
			max_vertex |= (uint64_t)(v0 | v1);
			if(max_weight < weight) max_weight = weight;
			(counts[edge_owner(v0,v1)])++;
			(counts[edge_owner(v1,v0)])++;
		} // #pragma omp for schedule(static)
	}

	// function #2
	template<typename EdgeType>
	void scanEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict counts, uint64_t& max_vertex, int& max_weight, typename EdgeType::no_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			max_vertex |= (uint64_t)(v0 | v1);
			(counts[edge_owner(v0,v1)])++;
			(counts[edge_owner(v1,v0)])++;
		} // #pragma omp for schedule(static)
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void reduceMaxWeight(int max_weight, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		int global_max_weight;
		MPI_Allreduce(&max_weight, &global_max_weight, 1, MPI_INT, MPI_MAX, mpi.comm_2d);
		g.max_weight_ = global_max_weight;
		g.log_max_weight_ = get_msb_index(global_max_weight);
	}

	// function #2
	template<typename EdgeType>
	void reduceMaxWeight(int max_weight, GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
	}

	void scatterAndScanEdges(EdgeList* edge_list, bool sort_by_degree, GraphType& g) {
		VT_TRACER("scan_edge");
		ScatterContext scatter(mpi.comm_2d);
		TwodVertex* edges_to_send = static_cast<TwodVertex*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(TwodVertex)));
		int num_loops = edge_list->beginRead(false);
		uint64_t max_vertex = 0;
		int max_weight = 0;

		if(mpi.isMaster()) fprintf(IMD_OUT, "Begin counting edges. Number of iterations is %d.\n", num_loops);

		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			EdgeType* edge_data;
			const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel reduction(|:max_vertex)
			{
				int* restrict counts = scatter.get_counts();
				scanEdges(edge_data, edge_data_length, counts, max_vertex, max_weight);
			} // #pragma omp parallel

			scatter.sum();

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) fprintf(IMD_OUT, "MPI_Allreduce...\n");
#endif

			MPI_Allreduce(MPI_IN_PLACE, &max_vertex, 1, MpiTypeOf<uint64_t>::type, MPI_BOR, mpi.comm_2d);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) fprintf(IMD_OUT, "OK! \n");
#endif

			const int log_max_vertex = get_msb_index(max_vertex) + 1;
			if(g.log_actual_global_verts_ == 0) {
				initializeParameters(log_max_vertex,
						edge_list->num_local_edges()*mpi.size_2d, sort_by_degree, g);
			}
			else if(log_max_vertex != g.log_actual_global_verts_) {
				// max vertex estimation failure
				if (mpi.isMaster() == 0) {
					fprintf(IMD_OUT, "Restarting because of change of log_max_vertex from %d"
							" to %d\n", g.log_actual_global_verts_, log_max_vertex);
				}

				free(wide_row_starts_); wide_row_starts_ = NULL;
				free(row_starts_sup_); row_starts_sup_ = NULL;

				break;
			}

			const int log_local_verts = log_local_verts_;
			const int log_r = get_msb_index(mpi.size_2dr);
			const int log_size = get_msb_index(mpi.size_2d);
			const int64_t cmask = cmask_;
#define SWIZZLE_VERTEX_SRC(c) (((c) >> log_size) | ((((c) & cmask) >> log_r) << log_local_verts))

#pragma omp parallel
			{
				int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					const int64_t v0_swizzled = SWIZZLE_VERTEX_SRC(v0);
					const int64_t v1_swizzled = SWIZZLE_VERTEX_SRC(v1);
					//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[edge_owner(v0,v1)])++] = v0_swizzled;
					//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[edge_owner(v1,v0)])++] = v1_swizzled;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) fprintf(IMD_OUT, "MPI_Alltoall...\n");
#endif

#undef SWIZZLE_VERTEX_SRC
			TwodVertex* recv_edges = scatter.scatter(edges_to_send);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) fprintf(IMD_OUT, "OK! \n");
#endif

			const int64_t num_recv_edges = scatter.get_recv_count();
			scanEdges(recv_edges, num_recv_edges, g);

			scatter.free(recv_edges);

			if(mpi.isMaster()) fprintf(IMD_OUT, "Iteration %d finished.\n", loop_count);
		}
		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		reduceMaxWeight<EdgeType>(max_weight, g);

		if(wide_row_starts_ != NULL) {
			if(mpi.isMaster()) fprintf(IMD_OUT, "Computing edge offset.\n");
			const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
			const int64_t src_region_length = num_local_verts * mpi.size_2dc;
			const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);
			for(int64_t i = 1; i < num_wide_rows; ++i) {
				wide_row_starts_[i+1] += wide_row_starts_[i];
			}
#ifndef NDEBUG
			if(mpi.isMaster()) fprintf(IMD_OUT, "Copying edge_counts for debugging.\n");
			memcpy(row_starts_sup_, wide_row_starts_, (num_wide_rows+1)*sizeof(row_starts_sup_[0]));
#endif
		}

		if(mpi.isMaster()) fprintf(IMD_OUT, "Finished scattering edges.\n");
	}

	void constructFromWideCSR(GraphType& g) {
		VT_TRACER("form_csr");
		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);
		const int64_t row_bitmap_length = std::max<int64_t>(1, src_region_length >> LOG_NBPE);
		const int64_t num_edge_blocks = std::max<int64_t>(1, src_region_length >> LOG_BFELL_SORT);
		const int64_t num_local_edges = wide_row_starts_[num_wide_rows];

		VERVOSE(if(mpi.isMaster()) fprintf(IMD_OUT, "num_local_verts %f M\nsrc_region_length %f M\n"
				"num_wide_rows %f M\nrow_bitmap_length %f M\n"
				"num_edge_blocks %f M\n",
				to_mega(num_local_verts), to_mega(src_region_length),
				to_mega(num_wide_rows), to_mega(row_bitmap_length), to_mega(num_edge_blocks)));

		// make row bitmap
		if(mpi.isMaster()) fprintf(IMD_OUT, "Allocating row bitmap.\n");
		g.row_bitmap_ = static_cast<BitmapType*>
			(cache_aligned_xmalloc(row_bitmap_length*sizeof(BitmapType)));
		memset(g.row_bitmap_, 0x00, row_bitmap_length*sizeof(BitmapType));

		if(mpi.isMaster()) fprintf(IMD_OUT, "Making row bitmap.\n");
#pragma omp parallel for
		for(int64_t part_base = 0; part_base < src_region_length; part_base += EDGE_PART_SIZE) {
			int64_t part_idx = part_base >> LOG_EDGE_PART_SIZE;
			int64_t bmp_off = part_base >> LOG_NBPE;
			for(int64_t i = wide_row_starts_[part_idx]; i < wide_row_starts_[part_idx+1]; ++i) {
				int64_t word_idx = (src_vertexes_[i] >> LOG_NBPE) + bmp_off;
				int bit_idx = src_vertexes_[i] & NBPE_MASK;
				g.row_bitmap_[word_idx] |= (BitmapType(1) << bit_idx);
			}
		}

		// make row sums
		g.row_sums_ = static_cast<TwodVertex*>
			(cache_aligned_xmalloc((row_bitmap_length+1)*sizeof(TwodVertex)));
		g.row_sums_[0] = 0;

		if(mpi.isMaster()) fprintf(IMD_OUT, "Making row sums bitmap.\n");
		for(int64_t i = 0; i < row_bitmap_length; ++i) {
			// TODO: deal with different BitmapType
#if USE_SPARC_ASM_POPC && FCC_OMP_ASM_BUG
			// With FCC, using inline assembler with openmp results broken code.
			int num_rows = sparc_popc_l_noinline(g.row_bitmap_[i]);
#else
			int num_rows = __builtin_popcountl(g.row_bitmap_[i]);
#endif
			g.row_sums_[i+1] = g.row_sums_[i] + num_rows;
		}

		const int64_t non_zero_rows = g.row_sums_[row_bitmap_length];
		int64_t* row_starts = static_cast<int64_t*>
			(cache_aligned_xmalloc((non_zero_rows+1)*sizeof(int64_t)));

		if(mpi.isMaster()) fprintf(IMD_OUT, "Computing row_starts.\n");
#pragma omp parallel for
		for(int64_t part_base = 0; part_base < src_region_length; part_base += EDGE_PART_SIZE) {
			int64_t part_idx = part_base >> LOG_EDGE_PART_SIZE;
			int64_t row_length[EDGE_PART_SIZE] = {0};
			int64_t edge_offset = wide_row_starts_[part_idx];
			for(int64_t i = wide_row_starts_[part_idx]; i < wide_row_starts_[part_idx+1]; ++i) {
				++(row_length[src_vertexes_[i] & EDGE_PART_SIZE_MASK]);
			}
			int part_end = (int)std::min<int64_t>(EDGE_PART_SIZE, src_region_length - part_base);
			for(int64_t i = 0; i < part_end; ++i) {
				int64_t word_idx = (part_base + i) >> LOG_NBPE;
				int bit_idx = i & NBPE_MASK;
				if(g.row_bitmap_[word_idx] & (BitmapType(1) << bit_idx)) {
					assert (row_length[i] > 0);
					BitmapType word = g.row_bitmap_[word_idx] & ((BitmapType(1) << bit_idx) - 1);
					TwodVertex row_offset = __builtin_popcountl(word) + g.row_sums_[word_idx];
					row_starts[row_offset] = edge_offset;
					edge_offset += row_length[i];
				}
				else {
					assert (row_length[i] == 0);
				}
			}
			assert (edge_offset == wide_row_starts_[part_idx+1]);
		}
		row_starts[non_zero_rows] = wide_row_starts_[num_wide_rows];

#ifndef NDEBUG
		// check row_starts
#pragma omp parallel for
		for(int64_t part_base = 0; part_base < src_region_length; part_base += EDGE_PART_SIZE) {
			int64_t part_idx = part_base >> LOG_EDGE_PART_SIZE;
			int64_t word_idx = part_base >> LOG_NBPE;
			int64_t nz_idx = g.row_sums_[word_idx];
			int num_rows = g.row_sums_[word_idx + EDGE_PART_SIZE / NBPE] - nz_idx;

			for(int i = 0; i < EDGE_PART_SIZE / NBPE; ++i) {
				int64_t diff = g.row_sums_[word_idx + i + 1] - g.row_sums_[word_idx + i];
				assert (diff == __builtin_popcountl(g.row_bitmap_[word_idx + i]));
			}

			assert (row_starts[nz_idx] == wide_row_starts_[part_idx]);
			for(int i = 0; i < num_rows; ++i) {
				assert (row_starts[nz_idx + i + 1] > row_starts[nz_idx + i]);
			}
		}
#endif // #ifndef NDEBUG

		// delete wide row structure
		free(wide_row_starts_); wide_row_starts_ = NULL;
		free(src_vertexes_); src_vertexes_ = NULL;

#if BFELL
		g.blk_off = static_cast<typename GraphType::BlockOffset*>
			(cache_aligned_xmalloc((num_edge_blocks+1)*sizeof(typename GraphType::BlockOffset)));
		g.sorted_idx_ = static_cast<SortIdx*>
			(cache_aligned_xmalloc(non_zero_rows*sizeof(SortIdx)));

		if(mpi.isMaster()) fprintf(IMD_OUT, "Constructing BFELL.\n");
		// compute column_length size
		int64_t col_len_naive = 0, col_len_opt = 0;
#pragma omp parallel
		{
			int64_t tmp_buffer_length = 2*1024;
			TwodVertex* restrict tmp_buffer =
					(TwodVertex*)cache_aligned_xmalloc(sizeof(TwodVertex)*tmp_buffer_length);

#pragma omp for reduction(+:col_len_naive,col_len_opt)
			for(int64_t blk_base = 0; blk_base < src_region_length; blk_base += BFELL_SORT) {
				int64_t word_idx = blk_base >> LOG_NBPE;
				int64_t row_length[BFELL_SORT] = {0};
				SortIdx row_map[BFELL_SORT];
				int64_t non_zero_offset = g.row_sums_[word_idx];
				int num_rows = g.row_sums_[word_idx + BFELL_SORT_IN_BMP] - non_zero_offset;
				assert (num_rows < BFELL_SORT);

				for(int i = 0; i < num_rows; ++i) {
					row_length[i] = row_starts[non_zero_offset+i+1] - row_starts[non_zero_offset+i];
					row_map[i]= i;
				}

				sort2(row_length, row_map, num_rows, std::greater<int64_t>());
				assert (row_length[0] >= row_length[1]);

				SortIdx* sorted_idx_ptr = g.sorted_idx_ + non_zero_offset;
				for(int i = 0; i < num_rows; ++i) {
#ifndef NDEBUG
				//	assert (row_map[i] < BFELL_SORT);
					int64_t row_idx = non_zero_offset + row_map[i];
					assert (row_length[i] == row_starts[row_idx + 1] - row_starts[row_idx]);
#endif
					sorted_idx_ptr[row_map[i]] = i;
				}

				int64_t blk_off = row_starts[non_zero_offset];
				int64_t block_length = row_starts[non_zero_offset + num_rows] - blk_off;
				if(block_length > tmp_buffer_length) {
					free(tmp_buffer);
					while(block_length > tmp_buffer_length) tmp_buffer_length *= 2;
					tmp_buffer = (TwodVertex*)cache_aligned_xmalloc(sizeof(TwodVertex)*tmp_buffer_length);
				}
				memcpy(tmp_buffer, g.edge_array_ + blk_off, block_length * sizeof(TwodVertex));

				int cur_num_rows = num_rows;
				int64_t column_idx = 0;
				int64_t edge_offset = 0;
				for( ; cur_num_rows > 0; --cur_num_rows) {
					while(row_length[cur_num_rows-1] > column_idx) {
						for(int i = 0; i < cur_num_rows; ++i) {
							g.edge_array_[blk_off + edge_offset + i] =
									tmp_buffer[row_starts[non_zero_offset + row_map[i]] - blk_off + column_idx];
						}
						edge_offset += cur_num_rows;
						++column_idx;
					}
				}
				assert (edge_offset == block_length);

				int64_t blk_idx = blk_base >> LOG_BFELL_SORT;
				col_len_naive += row_length[0] + 1;
				col_len_opt += row_length[1] + 1;
				g.blk_off[blk_idx+1].length_start = row_length[0] + 1;
				g.blk_off[blk_idx].edge_start = blk_off;
			} // #pragma omp for reduction(+:col_len_naive,col_len_opt)
			free(tmp_buffer);
		} // #pragma omp parallel

		if(mpi.isMaster()) fprintf(IMD_OUT, "Creating column length array.\n");
		g.col_len_ = static_cast<SortIdx*>
			(cache_aligned_xmalloc(col_len_naive*sizeof(SortIdx)));

		g.blk_off[0].length_start = 0;
		g.blk_off[num_edge_blocks].edge_start = row_starts[non_zero_rows];
		for(int64_t i = 1; i < num_edge_blocks; ++i) {
			g.blk_off[i+1].length_start += g.blk_off[i].length_start;
			assert (g.blk_off[i+1].edge_start >= g.blk_off[i].edge_start);
		}

#pragma omp parallel for
		for(int64_t blk_base = 0; blk_base < src_region_length; blk_base += BFELL_SORT) {
			int64_t word_idx = blk_base >> LOG_NBPE;
			int64_t row_length[BFELL_SORT];
			int64_t non_zero_offset = g.row_sums_[word_idx];
			int num_rows = g.row_sums_[word_idx + BFELL_SORT_IN_BMP] - non_zero_offset;

			for(int i = 0; i < num_rows; ++i) {
				row_length[g.sorted_idx_[non_zero_offset+i]] =
						row_starts[non_zero_offset+i+1] - row_starts[non_zero_offset+i];
			}

			int64_t blk_idx = blk_base >> LOG_BFELL_SORT;
			int64_t col_len_off = g.blk_off[blk_idx].length_start;
			int cur_num_rows = num_rows;
			int64_t column_idx = 0;
			int64_t edge_offset = 0;
			for( ; cur_num_rows > 0; --cur_num_rows) {
				while(row_length[cur_num_rows-1] > column_idx) {
					g.col_len_[col_len_off + column_idx] = cur_num_rows;
					edge_offset += cur_num_rows;
					++column_idx;
				}
			}
			g.col_len_[col_len_off + column_idx] = 0;
			assert (col_len_off + column_idx + 1 == g.blk_off[blk_idx + 1].length_start);

#ifndef NDEBUG
			int64_t blk_off = row_starts[non_zero_offset];
			int64_t block_length = row_starts[non_zero_offset + num_rows] - blk_off;
			assert (edge_offset == block_length);
#endif
		} // #pragma omp parallel for
		free(row_starts); row_starts = NULL;
#else // #if BFELL
		g.row_starts_ = row_starts;
#endif // #if BFELL

#if VERVOSE_MODE
		int64_t send_rowbmp[5] = { non_zero_rows, row_bitmap_length*NBPE, num_local_edges,
#if BFELL
				col_len_opt, col_len_naive
#else // #if BFELL
				0, 0
#endif // #if BFELL
		};
		int64_t max_rowbmp[5];
		int64_t sum_rowbmp[5];
		MPI_Reduce(send_rowbmp, sum_rowbmp, 5, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);
		MPI_Reduce(send_rowbmp, max_rowbmp, 5, MpiTypeOf<int64_t>::type, MPI_MAX, 0, mpi.comm_2d);
		if(mpi.isMaster()) {
			fprintf(IMD_OUT, "non zero rows. Total %f M / %f M = %f %% Avg %f M / %f M Max %f %%+\n",
					to_mega(sum_rowbmp[0]), to_mega(sum_rowbmp[1]), to_mega(sum_rowbmp[0]) / to_mega(sum_rowbmp[1]) * 100,
					to_mega(sum_rowbmp[0]) / mpi.size_2d, to_mega(sum_rowbmp[1]) / mpi.size_2d,
					diff_percent(max_rowbmp[0], send_rowbmp[0], mpi.size_2d));
			fprintf(IMD_OUT, "distributed edges. Total %f M Avg %f M Max %f %%+\n",
					to_mega(sum_rowbmp[2]), to_mega(sum_rowbmp[2]) / mpi.size_2d,
					diff_percent(max_rowbmp[2], sum_rowbmp[2], mpi.size_2d));
#if BFELL
			fprintf(IMD_OUT, "column_length size (opt/naive). Total %f M / %f M Avg %f M / %f M Max %f %%+ / %f %%+\n",
					to_mega(sum_rowbmp[3]), to_mega(sum_rowbmp[4]),
					to_mega(sum_rowbmp[3]) / mpi.size_2d, to_mega(sum_rowbmp[4]) / mpi.size_2d,
					diff_percent(max_rowbmp[3], sum_rowbmp[3], mpi.size_2d),
					diff_percent(max_rowbmp[4], sum_rowbmp[4], mpi.size_2d));
#endif
			fprintf(IMD_OUT, "Type requirements:\n");
			fprintf(IMD_OUT, "Global vertex id %s using %s\n", minimum_type(num_local_verts * mpi.size_2d), TypeName<int64_t>::value);
			fprintf(IMD_OUT, "Local vertex id %s using %s\n", minimum_type(num_local_verts), TypeName<uint32_t>::value);
			fprintf(IMD_OUT, "Index for local edges %s using %s\n", minimum_type(max_rowbmp[2]), TypeName<int64_t>::value);
			fprintf(IMD_OUT, "*Index for src local region %s using %s\n", minimum_type(num_local_verts * mpi.size_2dc), TypeName<TwodVertex>::value);
			fprintf(IMD_OUT, "*Index for dst local region %s using %s\n", minimum_type(num_local_verts * mpi.size_2dr), TypeName<TwodVertex>::value);
			fprintf(IMD_OUT, "Index for non zero rows %s using %s\n", minimum_type(max_rowbmp[0]), TypeName<TwodVertex>::value);
			fprintf(IMD_OUT, "*BFELL sort region size %s using %s\n", minimum_type(BFELL_SORT), TypeName<SortIdx>::value);
			fprintf(IMD_OUT, "Memory consumption:\n");
			fprintf(IMD_OUT, "row_bitmap %f MB\n", to_mega(row_bitmap_length*sizeof(BitmapType)));
			fprintf(IMD_OUT, "row_sums %f MB\n", to_mega((row_bitmap_length+1)*sizeof(TwodVertex)));
			fprintf(IMD_OUT, "edge_array %f MB\n", to_mega(max_rowbmp[2]*sizeof(TwodVertex)));
#if BFELL
			fprintf(IMD_OUT, "block_offset %f MB\n", to_mega((src_region_length/BFELL_SORT)*sizeof(TwodVertex)*2));
			fprintf(IMD_OUT, "sorted_idx %f MB\n", to_mega(max_rowbmp[0]*sizeof(SortIdx)));
			fprintf(IMD_OUT, "column_length(opt) %f MB\n", to_mega(max_rowbmp[3]*sizeof(SortIdx)));
			fprintf(IMD_OUT, "column_length(naive) %f MB\n", to_mega(max_rowbmp[4]*sizeof(SortIdx)));
#else
			fprintf(IMD_OUT, "row_starts %f MB\n", to_mega(row_bitmap_length*sizeof(BitmapType)));
#endif
		}
#endif // #if VERVOSE_MODE
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::has_weight dummy = 0)
	{
		const int log_local_verts = log_local_verts_;
		const int log_size = get_msb_index(mpi.size_2d);
		const int log_r = get_msb_index(mpi.size_2dr);
		const int64_t rmask = rmask_;
		const int64_t cmask = cmask_;
#define SWIZZLE_VERTEX_SRC(c) (((c) >> log_size) | ((((c) & cmask) >> log_r) << log_local_verts))
#define SWIZZLE_VERTEX_DST(c) (((c) >> log_size) | (((c) & rmask) << log_local_verts))
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(
					SWIZZLE_VERTEX_SRC(v0), SWIZZLE_VERTEX_DST(v1), edge_data[i].weight_);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(
					SWIZZLE_VERTEX_SRC(v1), SWIZZLE_VERTEX_DST(v0), edge_data[i].weight_);
		} // #pragma omp for schedule(static)
#undef SWIZZLE_VERTEX_SRC
#undef SWIZZLE_VERTEX_DST
	}

	// function #2
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::no_weight dummy = 0)
	{
		const int log_local_verts = log_local_verts_;
		const int log_size = get_msb_index(mpi.size_2d);
		const int log_r = get_msb_index(mpi.size_2dr);
		const int64_t rmask = rmask_;
		const int64_t cmask = cmask_;

#define SWIZZLE_VERTEX_SRC(c) (((c) >> log_size) | ((((c) & cmask) >> log_r) << log_local_verts))
#define SWIZZLE_VERTEX_DST(c) (((c) >> log_size) | (((c) & rmask) << log_local_verts))
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(SWIZZLE_VERTEX_SRC(v0), SWIZZLE_VERTEX_DST(v1));
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(SWIZZLE_VERTEX_SRC(v1), SWIZZLE_VERTEX_DST(v0));
		} // #pragma omp for schedule(static)
#undef SWIZZLE_VERTEX_SRC
#undef SWIZZLE_VERTEX_DST
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		const int log_local_src = log_local_verts_ + get_msb_index(mpi.size_2dc);
#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const int64_t v0 = edges[i].v0();
			const int64_t v1 = edges[i].v1();
			const int weight = edges[i].weight_;

			const int src_high = v0 >> LOG_EDGE_PART_SIZE;
			const uint16_t src_low = v0 & EDGE_PART_SIZE_MASK;
			const int64_t pos = __sync_fetch_and_add(&wide_row_starts_[src_high], 1);

			// random access (write)
#ifndef NDEBUG
			assert( g.edge_array_[pos] == 0 );
#endif
			src_vertexes_[pos] = src_low;
			g.edge_array_[pos] = (weight << log_local_src) | v1;
		}
	}

	// function #2
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const int64_t v0 = edges[i].v0();
			const int64_t v1 = edges[i].v1();

			const int src_high = v0 >> LOG_EDGE_PART_SIZE;
			const uint16_t src_low = v0 & EDGE_PART_SIZE_MASK;
			const int64_t pos = __sync_fetch_and_add(&wide_row_starts_[src_high], 1);

			// random access (write)
#ifndef NDEBUG
			assert( g.edge_array_[pos] == 0 );
#endif
			src_vertexes_[pos] = src_low;
			g.edge_array_[pos] = v1;
		}
	}

	void scatterAndStore(EdgeList* edge_list, bool sort_by_degree, GraphType& g) {
		VT_TRACER("store_edge");
		ScatterContext scatter(mpi.comm_2d);
		EdgeType* edges_to_send = static_cast<EdgeType*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(EdgeType)));

		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);
		g.edge_array_ = (TwodVertex*)malloc(wide_row_starts_[num_wide_rows]*sizeof(TwodVertex));
		src_vertexes_ = (uint16_t*)malloc(wide_row_starts_[num_wide_rows]*sizeof(uint16_t));

		int num_loops = edge_list->beginRead(true);

		if(mpi.isMaster()) fprintf(IMD_OUT, "Begin construction. Number of iterations is %d.\n", num_loops);

		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			EdgeType* edge_data;
			const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
			{
				int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					(counts[edge_owner(v0,v1)])++;
					(counts[edge_owner(v1,v0)])++;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

			scatter.sum();

#pragma omp parallel
			{
				int* offsets = scatter.get_offsets();
				writeSendEdges(edge_data, edge_data_length, offsets, edges_to_send);
			}

			if(mpi.isMaster()) fprintf(IMD_OUT, "Scatter edges.\n");

			EdgeType* recv_edges = scatter.scatter(edges_to_send);
			const int num_recv_edges = scatter.get_recv_count();

			if(mpi.isMaster()) fprintf(IMD_OUT, "Add edges.\n");

			addEdges(recv_edges, num_recv_edges, g);

			scatter.free(recv_edges);

			if(mpi.isMaster()) fprintf(IMD_OUT, "Iteration %d finished.\n", loop_count);
		}

		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		if(mpi.isMaster()) fprintf(IMD_OUT, "Refreshing edge offset.\n");
		memmove(wide_row_starts_+1, wide_row_starts_, num_wide_rows*sizeof(wide_row_starts_[0]));
		wide_row_starts_[0] = 0;

#ifndef NDEBUG
#pragma omp parallel for
		for(int64_t i = 0; i <= num_wide_rows; ++i) {
			if(row_starts_sup_[i] != wide_row_starts_[i]) {
				fprintf(IMD_OUT, "Error: Edge Counts: i=%"PRId64",1st=%"PRId64",2nd=%"PRId64"\n", i, row_starts_sup_[i], wide_row_starts_[i]);
			}
			assert(row_starts_sup_[i] == wide_row_starts_[i]);
		}
#endif
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void sortEdgesInner(GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		/*
		int64_t sort_buffer_length = 2*1024;
		int64_t* restrict sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);

		const int64_t num_edge_lists = (int64_t(1) << g.log_edge_lists());
		const int log_weight_bits = g.log_packing_edge_lists_;
		const int log_packing_edge_lists = g.log_packing_edge_lists();
		const int index_bits = g.log_global_verts() - get_msb_index(mpi.size_2dr);
		const int64_t mask_packing_edge_lists = (int64_t(1) << log_packing_edge_lists) - 1;
		const int64_t mask_weight = (int64_t(1) << log_weight_bits) - 1;
		const int64_t mask_index = (int64_t(1) << index_bits) - 1;
		const int64_t mask_index_compare =
				(mask_index << (log_packing_edge_lists + log_weight_bits)) |
				mask_packing_edge_lists;

#define ENCODE(v) \
		(((((v & mask_packing_edge_lists) << log_weight_bits) | \
		((v >> log_packing_edge_lists) & mask_weight)) << index_bits) | \
		(v >> (log_packing_edge_lists + log_weight_bits)))
#define DECODE(v) \
		(((((v & mask_index) << log_weight_bits) | \
		((v >> index_bits) & mask_weight)) << log_packing_edge_lists) | \
		(v >> (index_bits + log_weight_bits)))

#pragma omp for
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			const int64_t edge_count = wide_row_starts_[i];
			const int64_t rowstart_i = g.row_starts_[i];
			assert (g.row_starts_[i+1] - g.row_starts_[i] == wide_row_starts_[i]);

			if(edge_count > sort_buffer_length) {
				free(sort_buffer);
				while(edge_count > sort_buffer_length) sort_buffer_length *= 2;
				sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
			}

			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t v = g.edge_array_(rowstart_i + c);
				sort_buffer[c] = ENCODE(v);
				assert(v == DECODE(ENCODE(v)));
			}
			// sort sort_buffer
			std::sort(sort_buffer, sort_buffer + edge_count);

			int64_t idx = rowstart_i;
			int64_t prev_v = -1;
			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t sort_v = sort_buffer[c];
				// TODO: now duplicated edges are not merged because sort order is
				// v0 row bits > weight > index
				// To reduce parallel edges, sort by the order of
				// v0 row bits > index > weight
				// and if you want to optimize SSSP, sort again by the order of
				// v0 row bits > weight > index
			//	if((prev_v & mask_index_compare) != (sort_v & mask_index_compare)) {
					assert (prev_v < sort_v);
					const int64_t v = DECODE(sort_v);
					g.edge_array_.set(idx, v);
			//		prev_v = sort_v;
					idx++;
			//	}
			}
		//	if(wide_row_starts_[i] > idx - rowstart_i) {
				wide_row_starts_[i] = idx - rowstart_i;
		//	}
		} // #pragma omp for

#undef ENCODE
#undef DECODE
		free(sort_buffer);
*/
	}

	struct SortEdgeCompair {
		typedef pointer_pair_value<uint16_t, TwodVertex> Val;
		bool operator ()(Val r1, Val r2) const {
			uint64_t r1_v = (uint64_t(r1.v1) << 48) | r1.v2;
			uint64_t r2_v = (uint64_t(r2.v1) << 48) | r2.v2;
			return r1_v < r2_v;
		}
	};

	// function #2
	template<typename EdgeType>
	void sortEdgesInner(GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);


#pragma omp for
		for(int64_t i = 0; i < num_wide_rows; ++i) {
			const int64_t edge_offset = wide_row_starts_[i];
			const int64_t edge_count = wide_row_starts_[i+1] - edge_offset;

			// sort
			sort2(src_vertexes_ + edge_offset, g.edge_array_ + edge_offset, edge_count, SortEdgeCompair());

			int64_t idx = edge_offset;
			uint64_t prev_v = (uint64_t(src_vertexes_[idx]) << 48) | g.edge_array_[idx]; ++idx;
			for(int64_t c = edge_offset+1; c < edge_offset + edge_count; ++c) {
				const uint64_t sort_v = (uint64_t(src_vertexes_[c]) << 48) | g.edge_array_[c];
				if(prev_v != sort_v) {
					assert (prev_v < sort_v);
					g.edge_array_[idx] = g.edge_array_[c];
					src_vertexes_[idx] = src_vertexes_[c];
					prev_v = sort_v;
					++idx;
				}
			}
			row_starts_sup_[i] = idx - edge_offset;
		} // #pragma omp for

#undef ENCODE
#undef DECODE
	}

	void sortEdges(GraphType& g) {
		VT_TRACER("sort_edge");
		if(mpi.isMaster()) fprintf(IMD_OUT, "Sorting edges.\n");

#pragma omp parallel
		sortEdgesInner<EdgeType>(g);

		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);
		// this loop can't be parallel
		int64_t rowstart_new = 0;
		for(int64_t i = 0; i < num_wide_rows; ++i) {
			const int64_t edge_count_new = row_starts_sup_[i];
			const int64_t rowstart_old = wide_row_starts_[i]; // read before write
			wide_row_starts_[i] = rowstart_new;
			if(rowstart_new != rowstart_old) {
				memmove(src_vertexes_ + rowstart_new, src_vertexes_ + rowstart_old, edge_count_new * sizeof(src_vertexes_[0]));
				memmove(g.edge_array_ + rowstart_new, g.edge_array_ + rowstart_old, edge_count_new * sizeof(g.edge_array_[0]));
			}
			rowstart_new += edge_count_new;
		}
		const int64_t old_num_edges = wide_row_starts_[num_wide_rows];
		wide_row_starts_[num_wide_rows] = rowstart_new;

		 int64_t num_edge_sum[2] = {0};
		 int64_t num_edge[2] = {old_num_edges, rowstart_new};
		MPI_Reduce(num_edge, num_edge_sum, 2, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);
		if(mpi.isMaster()) fprintf(IMD_OUT, "# of edges is reduced. Total %zd -> %zd Diff %f %%\n",
				num_edge_sum[0], num_edge_sum[1], (double)(num_edge_sum[0] - num_edge_sum[1])/(double)num_edge_sum[0]*100.0);
		g.num_global_edges_ = num_edge_sum[1];
	}

	void computeNumVertices(GraphType& g) {
		VT_TRACER("num_verts");
		const int local_bitmap_width = (int64_t(1) << g.log_local_bitmap());
		int recvcounts[mpi.size_2dc];
		for(int i = 0; i < mpi.size_2dc; ++i) recvcounts[i] = local_bitmap_width;

		g.has_edge_bitmap_ = (BitmapType*)cache_aligned_xmalloc(local_bitmap_width*sizeof(BitmapType));
		MPI_Reduce_scatter(g.row_bitmap_, g.has_edge_bitmap_, recvcounts, MpiTypeOf<BitmapType>::type, MPI_BOR, mpi.comm_2dr);
		int64_t num_vertices = 0;
#pragma omp parallel for reduction(+:num_vertices)
		for(int i = 0; i < local_bitmap_width; ++i) {
			num_vertices += __builtin_popcountl(g.has_edge_bitmap_[i]);
		}
		MPI_Allreduce(MPI_IN_PLACE, &num_vertices, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
		VERVOSE(int64_t num_virtual_vertices = int64_t(1) << g.log_actual_global_verts_);
		VERVOSE(if(mpi.isMaster()) fprintf(IMD_OUT, "# of actual vertices %f G %f %%\n", to_giga(num_vertices),
				(double)num_vertices / (double)num_virtual_vertices * 100.0));
		g.num_global_verts_ = num_vertices;
	}

	const int log_size_;
	const int rmask_;
	const int cmask_;
	int log_minimum_global_verts_;
	int log_local_verts_;

	uint16_t* src_vertexes_;
	int64_t* wide_row_starts_;
	int64_t* row_starts_sup_;
};

} // namespace detail {

template <typename EdgeList>
void construct_graph(EdgeList* edge_list, bool sort_by_degree, bool enable_folding,
		Graph2DCSR& g)
{
	detail::GraphConstructor2DCSR<EdgeList> constructor;
	constructor.construct(edge_list, sort_by_degree, enable_folding, g);
}


#endif /* GRAPH_CONSTRUCTOR_HPP_ */
