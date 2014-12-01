/*
 * graph_constructor.hpp
 *
 *  Created on: Dec 14, 2011
 *      Author: koji
 */

#ifndef GRAPH_CONSTRUCTOR_HPP_
#define GRAPH_CONSTRUCTOR_HPP_

#include "parameters.h"
#include "limits.h"

//-------------------------------------------------------------//
// 2D partitioning
//-------------------------------------------------------------//

int inline vertex_owner_r(int64_t v) { return v % mpi.size_2dr; }
int inline vertex_owner_c(int64_t v) { return (v / mpi.size_2dr) % mpi.size_2dc; }
int inline edge_owner(int64_t v0, int64_t v1) { return vertex_owner_r(v0) + vertex_owner_c(v1) * mpi.size_2dr; }
int inline vertex_owner(int64_t v) { return v % mpi.size_2d; }
int64_t inline vertex_local(int64_t v) { return v / mpi.size_2d; }

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
	, has_edge_bitmap_(NULL)
#if BFELL
	, blk_off(NULL)
	, sorted_idx_(NULL)
	, col_len_(NULL)
#else
	, row_starts_(NULL)
	, degree_(NULL)
	, isolated_edges_(NULL)
#endif
	, log_actual_global_verts_(0)
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
		free(has_edge_bitmap_); has_edge_bitmap_ = NULL;
		free(edge_array_); edge_array_ = NULL;
#if BFELL
		free(sorted_idx_); sorted_idx_ = NULL;
		free(col_len_); col_len_ = NULL;
		free(blk_off); blk_off = NULL;
#else
		free(row_starts_); row_starts_ = NULL;
#if ISOLATE_FIRST_EDGE
		free(isolated_edges_); isolated_edges_ = NULL;
#endif // #if ISOLATE_FIRST_EDGE
#endif
	}

	int log_actual_global_verts() const { return log_actual_global_verts_; }

	// Reference Functions
	static int rank(int r, int c) { return c * mpi.size_2dr + r; }
	int64_t swizzle_vertex(int64_t v) {
		return SeparatedId(vertex_owner(v), vertex_local(v), local_bits_).value;
	}
	int64_t unswizzle_vertex(int64_t v) {
		SeparatedId id(v);
		return id.high(local_bits_) + id.low(local_bits_) * num_local_verts_;
	}

	// vertex id converter
	SeparatedId VtoD(int64_t v) {
		return SeparatedId(vertex_owner_r(v), vertex_local(v), local_bits_);
	}
	SeparatedId VtoS(int64_t v) {
		return SeparatedId(vertex_owner_c(v), vertex_local(v), local_bits_);
	}
	int64_t DtoV(SeparatedId id, int c) {
		return id.low(local_bits_) * mpi.size_2d + rank(id.high(local_bits_), c);
	}
	int64_t StoV(SeparatedId id, int r) {
		return id.low(local_bits_) * mpi.size_2d + rank(r, id.high(local_bits_));
	}
	SeparatedId StoD(SeparatedId id, int r) {
		return SeparatedId(r, id.low(local_bits_), local_bits_);
	}
	SeparatedId DtoS(SeparatedId id, int c) {
		return SeparatedId(c, id.low(local_bits_), local_bits_);
	}
	int get_weight_from_edge(int64_t e) {
		return e & ((1 << log_max_weight_) - 1);
	}

	bool has_edge(int64_t v, bool has_weight = false) {
		if(vertex_owner(v) == mpi.rank_2d) {
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
	int64_t* degree_;
	TwodVertex* isolated_edges_;
#endif

	int log_actual_global_verts_; // estimated SCALE parameter
	int log_max_weight_;

	int max_weight_;
	int64_t num_global_edges_;
	int64_t num_global_verts_;

	int local_bits_;
	int64_t num_local_verts_;
};

namespace detail {

template <typename EdgeList>
class GraphConstructor2DCSR
{
	enum {
		LOG_EDGE_PART_SIZE = 16,
	//	LOG_EDGE_PART_SIZE = 12,
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
		: src_vertexes_(NULL)
		, wide_row_starts_(NULL)
	{ }
	~GraphConstructor2DCSR()
	{
		// since the heap checker of FUJITSU compiler reports error on free(NULL) ...
		if(src_vertexes_ != NULL) { free(src_vertexes_); src_vertexes_ = NULL; }
		if(wide_row_starts_ != NULL) { free(wide_row_starts_); wide_row_starts_ = NULL; }
	}

	void construct(EdgeList* edge_list, int log_local_verts_unit, GraphType& g)
	{
		TRACER(construction);
		log_local_verts_unit_ = log_local_verts_unit;
		g.log_actual_global_verts_ = 0;

		do { // loop for max vertex estimation failure
			scatterAndScanEdges(edge_list, g);
		} while(g.log_actual_global_verts_ == 0);

		scatterAndStore(edge_list, g);
		sortEdges(g);
		free(row_starts_sup_); row_starts_sup_ = NULL;

		if(mpi.isMaster()) print_with_prefix("Wide CSR creation complete.");

		constructFromWideCSR(g);

		computeNumVertices(g);

		if(mpi.isMaster()) print_with_prefix("Graph construction complete.");
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

	void initializeParameters(
		int log_max_vertex,
		int64_t num_global_edges,
		GraphType& g)
	{
		int64_t num_global_verts = int64_t(1) << log_max_vertex;
		int64_t local_verts_unit = int64_t(1) << log_local_verts_unit_;
		int64_t num_local_verts = roundup(num_global_verts / mpi.size_2d, local_verts_unit);

		g.log_actual_global_verts_ = log_max_vertex;
		g.num_local_verts_ = num_local_verts;
		g.num_global_verts_ = num_local_verts * mpi.size_2d;
		local_bits_ = g.local_bits_ = get_msb_index(num_local_verts - 1) + 1;

		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		num_wide_rows_ = (src_region_length + EDGE_PART_SIZE - 1) / EDGE_PART_SIZE;

		wide_row_starts_ = static_cast<int64_t*>
			(cache_aligned_xmalloc((num_wide_rows_+1)*sizeof(wide_row_starts_[0])));
		memset(wide_row_starts_, 0x00, (num_wide_rows_+1)*sizeof(wide_row_starts_[0]));
		row_starts_sup_ = static_cast<int64_t*>(
				cache_aligned_xmalloc((num_wide_rows_+1)*sizeof(row_starts_sup_[0])));
	}

	void scanEdges(TwodVertex* edges, int64_t num_edges, GraphType& g) {
		int lgl = g.local_bits_;
		int L = g.num_local_verts_;
#define EDGE_PART_IDX(v) ((SeparatedId(v).compact(lgl, L) >> LOG_EDGE_PART_SIZE) + 1)
		int64_t i;
#pragma omp parallel for schedule(static)
		for(i = 0; i < (num_edges&(~3)); i += 4) {
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+0])], 1);
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+1])], 1);
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+2])], 1);
			__sync_fetch_and_add(&wide_row_starts_[EDGE_PART_IDX(edges[i+3])], 1);
		}
		for(i = (num_edges&(~3)); i < num_edges; ++i) {
			wide_row_starts_[EDGE_PART_IDX(edges[i])]++;
		}
#undef EDGE_PART_IDX
	}

	class DegreeConverter
	{
	public:
		typedef TwodVertex send_type;
		typedef int64_t recv_type;

		DegreeConverter(TwodVertex* edges, int64_t* degree, int64_t* recv_degree, int local_bits)
			: edges_(edges)
			, degree_(degree)
			, recv_degree_(recv_degree)
			, local_bits_(local_bits)
		{ }
		int target(int i) const {
			const SeparatedId v1_swizzled(edges_[i]);
			assert (v1_swizzled.high(local_bits_) < mpi.size_2dr);
			return v1_swizzled.high(local_bits_);
		}
		TwodVertex get(int i) const {
			return SeparatedId(edges_[i]).low(local_bits_);
		}
		int64_t map(TwodVertex v) const {
			return degree_[v];
		}
		void set(int i, int64_t d) const {
			recv_degree_[i] = d;
		}
	private:
		const TwodVertex* const edges_;
		int64_t* degree_;
		int64_t* recv_degree_;
		const int local_bits_;
		const TwodVertex local_verts_mask_;
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

	void scatterAndScanEdges(EdgeList* edge_list, GraphType& g) {
		TRACER(scan_edge);
		ScatterContext scatter(mpi.comm_2d);
		TwodVertex* edges_to_send = static_cast<TwodVertex*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(TwodVertex)));
		int num_loops = edge_list->beginRead(false);
		uint64_t max_vertex = 0;
		int max_weight = 0;

		if(mpi.isMaster()) print_with_prefix("Begin counting edges. Number of iterations is %d.", num_loops);

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
			if(mpi.isMaster()) print_with_prefix("MPI_Allreduce...");
#endif

			uint64_t tmp_send = max_vertex;
			MPI_Allreduce(&tmp_send, &max_vertex, 1, MpiTypeOf<uint64_t>::type, MPI_BOR, mpi.comm_2d);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("OK! ");
#endif

			const int log_max_vertex = get_msb_index(max_vertex) + 1;
			if(g.log_actual_global_verts_ == 0) {
				initializeParameters(log_max_vertex,
						edge_list->num_local_edges()*mpi.size_2d, g);
			}
			else if(log_max_vertex != g.log_actual_global_verts_) {
				// max vertex estimation failure
				if (mpi.isMaster() == 0) {
					print_with_prefix("Restarting because of change of log_max_vertex from %d"
							" to %d\n", g.log_actual_global_verts_, log_max_vertex);
				}

				free(wide_row_starts_); wide_row_starts_ = NULL;
				free(row_starts_sup_); row_starts_sup_ = NULL;

				break;
			}

			const int local_bits = local_bits_;
#pragma omp parallel
			{
				int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					const SeparatedId v0_swizzled(vertex_owner_c(v0), vertex_local(v0), local_bits);
					const SeparatedId v1_swizzled(vertex_owner_c(v1), vertex_local(v1), local_bits);
					//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[edge_owner(v0,v1)])++] = v0_swizzled.value;
					//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[edge_owner(v1,v0)])++] = v1_swizzled.value;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("MPI_Alltoall...");
#endif

			TwodVertex* recv_edges = scatter.scatter(edges_to_send);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("OK! ");
#endif

			const int64_t num_recv_edges = scatter.get_recv_count();
			scanEdges(recv_edges, num_recv_edges, g);

			scatter.free(recv_edges);

			if(mpi.isMaster()) print_with_prefix("Iteration %d finished.", loop_count);
		}
		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		reduceMaxWeight<EdgeType>(max_weight, g);

		if(wide_row_starts_ != NULL) {
			if(mpi.isMaster()) print_with_prefix("Computing edge offset.");
			for(int64_t i = 1; i < num_wide_rows_; ++i) {
				wide_row_starts_[i+1] += wide_row_starts_[i];
			}
#ifndef NDEBUG
			if(mpi.isMaster()) print_with_prefix("Copying edge_counts for debugging.");
			memcpy(row_starts_sup_, wide_row_starts_, (num_wide_rows_+1)*sizeof(row_starts_sup_[0]));
#endif
		}

		if(mpi.isMaster()) print_with_prefix("Finished scattering edges.");
	}
#if !BFELL
	int64_t* computeGlobalDegree(GraphType& g) {
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t local_bitmap_width = num_local_verts / NBPE;
		const int64_t row_bitmap_length = local_bitmap_width * mpi.size_2dc;
		const int comm_size = mpi.size_2dc;
		int send_counts[comm_size], send_offsets[comm_size + 1];
		int recv_counts[comm_size], recv_offsets[comm_size + 1];
		const int num_loops = get_blocks(row_bitmap_length, EdgeList::CHUNK_SIZE / NBPE * 2);
		const int64_t bitmap_chunk_size = roundup<int64_t>(local_bitmap_width, num_loops);
		assert (bitmap_chunk_size > 0);
		int64_t *degree = static_cast<int64_t*>(
				cache_aligned_xmalloc(num_local_verts*sizeof(int64_t)));
		BitmapType *send_bmp = static_cast<BitmapType*>(
				page_aligned_xmalloc(bitmap_chunk_size * comm_size*sizeof(BitmapType)));
		BitmapType *recv_bmp = static_cast<BitmapType*>(
				page_aligned_xmalloc(bitmap_chunk_size * comm_size*sizeof(BitmapType)));
		int64_t *recv_row_sums = static_cast<int64_t*>(
				cache_aligned_xmalloc((bitmap_chunk_size + 1) * comm_size*sizeof(int64_t)));

		if(mpi.isMaster()) print_with_prefix("Start computing global degree. %d iterations.", num_loops);
		for(int loop = 0; loop < num_loops; ++loop) {
			TwodVertex local_start = std::min<TwodVertex>(bitmap_chunk_size * loop, local_bitmap_width);
			TwodVertex local_end = std::min<TwodVertex>(local_start + bitmap_chunk_size, local_bitmap_width);
			TwodVertex local_length = local_end - local_start;

			// compute count and offset
			send_offsets[0] = 0;
			for(int c = 0; c < mpi.size_2dc; ++c) {
				TwodVertex bmp_start = c * local_bitmap_width + local_start;
				send_counts[c] = g.row_sums_[bmp_start + local_length] - g.row_sums_[bmp_start];
				send_offsets[c + 1] = send_counts[c] + send_offsets[c];
			}

			int64_t *send_degree = static_cast<int64_t*>(
					cache_aligned_xmalloc(send_offsets[mpi.size_2dc]*sizeof(int64_t)));

			// copy data to send memory
#pragma omp parallel for
			for(int c = 0; c < comm_size; ++c) {
				TwodVertex bmp_start = c * local_bitmap_width + local_start;
				memcpy(send_bmp + local_length * c,
						g.row_bitmap_ + bmp_start, local_length * sizeof(BitmapType));
				int count_c = send_counts[c];
				for(int64_t r = 0; r < count_c; ++r) {
					send_degree[send_offsets[c] + r] = g.row_starts_[g.row_sums_[bmp_start] + r + 1]
					                           - g.row_starts_[g.row_sums_[bmp_start] + r];
				}
			}

			// transfer data
			MpiCol::alltoall(send_bmp, recv_bmp, local_length, mpi.comm_2dr);
			int64_t* recv_degree = MpiCol::alltoallv(send_degree, send_counts, send_offsets,
					recv_counts, recv_offsets, mpi.comm_2dr, comm_size);
			free(send_degree); send_degree = NULL;

			// create recv_row_sums
#pragma omp parallel for
			for(int c = 0; c < comm_size; ++c) {
				int64_t sum = 0;
				for(int64_t i = 0; i < local_length; ++i) {
					recv_row_sums[i + (local_length + 1) * c] = sum;
					sum += __builtin_popcountl(recv_bmp[i + local_length * c]);
				}
				recv_row_sums[local_length + (local_length + 1) * c] = sum;
			}

			// count degree
#pragma omp parallel for
			for(int64_t word_idx = 0; word_idx < local_length; ++word_idx) {
				int64_t cnt[NBPE] = {0};
				for(int c = 0; c < comm_size; ++c) {
					BitmapType row_bitmap_i = recv_bmp[word_idx + local_length * c];
					BitmapType bit_flags = row_bitmap_i;
					while(bit_flags != BitmapType(0)) {
						BitmapType vis_bit, mask; int bit_idx;
						NEXT_BIT(bit_flags, vis_bit, mask, bit_idx);
						TwodVertex non_zero_idx = recv_row_sums[word_idx] + __builtin_popcountl(row_bitmap_i & mask);
						cnt[bit_idx] += recv_degree[non_zero_idx];
					}
				}
				for(int i = 0; i < NBPE; ++i) {
					degree[(local_length * loop + word_idx) * NBPE + i] = cnt[i];
				}
			}

			MPI_Free_mem(recv_degree); recv_degree = NULL;
			if(mpi.isMaster()) print_with_prefix("%d-th iteration finished.", loop);
		} // for(int loop = 0; loop < num_loops; ++loop) {

		free(send_bmp); send_bmp = NULL;
		free(recv_bmp); recv_bmp = NULL;
		free(recv_row_sums); recv_row_sums = NULL;
		if(mpi.isMaster()) print_with_prefix("Finished computing global degree.");
		return degree;
	}

	void isolateFirstEdge(GraphType& g) {
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t local_bitmap_width = num_local_verts / NBPE;
		const int64_t row_bitmap_length = local_bitmap_width * mpi.size_2dc;
		const int64_t non_zero_rows = g.row_sums_[row_bitmap_length];
#if ISOLATE_FIRST_EDGE
		g.isolated_edges_ = static_cast<TwodVertex*>(
				cache_aligned_xmalloc(non_zero_rows*sizeof(g.isolated_edges_[0])));
#endif
		//const int comm_size = mpi.size_2dc;
		int64_t num_max_edges = g.row_starts_[non_zero_rows];
		int64_t tmp_send_num_max_edges = num_max_edges;
		MPI_Allreduce(&tmp_send_num_max_edges, &num_max_edges, 1, MpiTypeOf<int64_t>::type, MPI_MAX, mpi.comm_2d);
		int num_loops = get_blocks<int64_t>(num_max_edges, EdgeList::CHUNK_SIZE*4);
		int64_t bitmap_chunk_size = (num_loops == 0) ? 0 : get_blocks<int64_t>(row_bitmap_length, num_loops);

		if(mpi.isMaster()) print_with_prefix("Start sorting by degree. %d iterations.", num_loops);
		for(int loop = 0; loop < num_loops; ++loop) {
			int64_t bmp_start = std::min<TwodVertex>(row_bitmap_length, bitmap_chunk_size * loop);
			int64_t bmp_end = std::min<TwodVertex>(row_bitmap_length, bmp_start + bitmap_chunk_size);
			int64_t non_zer_row_start = g.row_sums_[bmp_start];
			int64_t non_zer_row_end = g.row_sums_[bmp_end];
#if DEGREE_ORDER
			int64_t edge_start = g.row_starts_[non_zer_row_start];
			int64_t edge_end = g.row_starts_[non_zer_row_end];
			int64_t length = edge_end - edge_start;
			int64_t* recv_degree = static_cast<int64_t*>(
					cache_aligned_xmalloc(length*sizeof(int64_t)));
			if(length > INT_MAX) {
				print_with_prefix("Error: Integer overflow (%s:%d)", __FILE__, __LINE__);
			}
			MpiCol::gather(DegreeConverter(g.edge_array_ + edge_start, g.degree_, recv_degree, g.log_local_verts()),
					length, mpi.comm_2dc);
#endif

			// sort edges by their degree.
#pragma omp for
			for(int64_t non_zero_idx = non_zer_row_start; non_zero_idx < non_zer_row_end; ++non_zero_idx) {
				int64_t e_start = g.row_starts_[non_zero_idx];
#if DEGREE_ORDER_ONLY_IE
				int64_t e_end = g.row_starts_[non_zero_idx + 1];
				int64_t* origin_degree = recv_degree - edge_start;
				int64_t pos = std::max_element(origin_degree + e_start, origin_degree + e_end) - origin_degree;
				g.isolated_edges_[non_zero_idx] = g.edge_array_[pos];
				memmove(g.edge_array_ + e_start + 1, g.edge_array_ + e_start, (pos - e_start)*sizeof(TwodVertex));
#else // #if DEGREE_ORDER_ONLY_IE
#if DEGREE_ORDER
				int64_t e_end = g.row_starts_[non_zero_idx + 1];
				sort2(recv_degree + e_start - edge_start, g.edge_array_ + e_start,
						e_end - e_start, std::greater<int64_t>());
#endif
#if ISOLATE_FIRST_EDGE
				g.isolated_edges_[non_zero_idx] = g.edge_array_[e_start];
#endif
#endif // #if DEGREE_ORDER_ONLY_IE
			} // #pragma omp for

#if ISOLATE_FIRST_EDGE
			// compact edge array
			// This loop cannot be parallelized.
			for(int64_t non_zero_idx = non_zer_row_start; non_zero_idx < non_zer_row_end; ++non_zero_idx) {
				int64_t e_start = g.row_starts_[non_zero_idx];
				int64_t e_end = g.row_starts_[non_zero_idx + 1];
				int64_t e_length = e_end - e_start;
				memmove(g.edge_array_ + e_start - non_zero_idx, g.edge_array_ + e_start + 1, sizeof(TwodVertex) * (e_length - 1));
				g.row_starts_[non_zero_idx] -= non_zero_idx;
			}
#endif
#if DEGREE_ORDER
			free(recv_degree); recv_degree = NULL;
#endif
			MPI_Barrier(mpi.comm_2d);
			if(mpi.isMaster()) print_with_prefix("%d-th iteration finished.", loop);
		} // for(int loop = 0; loop < num_loops; ++loop) {

#if ISOLATE_FIRST_EDGE
		// update the last entry of row_starts_ and compact the edge array memory
		g.row_starts_[non_zero_rows] -= non_zero_rows;
		g.edge_array_ = static_cast<TwodVertex*>(realloc(g.edge_array_,
				g.row_starts_[non_zero_rows] * sizeof(TwodVertex)));
		if(g.row_starts_[non_zero_rows] != 0 && g.edge_array_ == NULL) {
			throw_exception("Out of memory trying to re-allocate edge array");
		}
#endif

		if(mpi.isMaster()) print_with_prefix("Finished sorting by degree.");
	}
#endif
	void constructFromWideCSR(GraphType& g) {
		TRACER(form_csr);
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t row_bitmap_length = src_region_length >> LOG_NBPE;
		VERVOSE(const int64_t num_local_edges = wide_row_starts_[num_wide_rows_]);

		VERVOSE(if(mpi.isMaster()) {
			print_with_prefix("num_local_verts %f M", to_mega(num_local_verts));
			print_with_prefix("src_region_length %f M", to_mega(src_region_length));
			print_with_prefix("num_wide_rows %f M", to_mega(num_wide_rows_));
			print_with_prefix("row_bitmap_length %f M", to_mega(row_bitmap_length));
			print_with_prefix("local_bits=%d", local_bits_);
			print_with_prefix("correspond to %f M", to_mega(int64_t(1) << local_bits_));
		});

		// make row bitmap
		if(mpi.isMaster()) print_with_prefix("Allocating row bitmap.");
		g.row_bitmap_ = static_cast<BitmapType*>
			(cache_aligned_xmalloc(row_bitmap_length*sizeof(BitmapType)));
		memset(g.row_bitmap_, 0x00, row_bitmap_length*sizeof(BitmapType));

		if(mpi.isMaster()) print_with_prefix("Making row bitmap.");
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

		if(mpi.isMaster()) print_with_prefix("Making row sums bitmap.");
		for(int64_t i = 0; i < row_bitmap_length; ++i) {
			// TODO: deal with different BitmapType
			int num_rows = __builtin_popcountl(g.row_bitmap_[i]);
			g.row_sums_[i+1] = g.row_sums_[i] + num_rows;
		}

		const int64_t non_zero_rows = g.row_sums_[row_bitmap_length];
		int64_t* row_starts = static_cast<int64_t*>
			(cache_aligned_xmalloc((non_zero_rows+1)*sizeof(int64_t)));

		if(mpi.isMaster()) print_with_prefix("Computing row_starts.");
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
		row_starts[non_zero_rows] = wide_row_starts_[num_wide_rows_];

#ifndef NDEBUG
		// check row_starts
#pragma omp parallel for
		for(int64_t part_base = 0; part_base < src_region_length; part_base += EDGE_PART_SIZE) {
			int64_t part_size = std::min<int64_t>(src_region_length - part_base, EDGE_PART_SIZE);
			int64_t part_idx = part_base >> LOG_EDGE_PART_SIZE;
			int64_t word_idx = part_base >> LOG_NBPE;
			int64_t nz_idx = g.row_sums_[word_idx];
			int num_rows = g.row_sums_[word_idx + part_size / NBPE] - nz_idx;

			for(int i = 0; i < part_size / NBPE; ++i) {
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

		if(mpi.isMaster()) print_with_prefix("Constructing BFELL.");
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

		if(mpi.isMaster()) print_with_prefix("Creating column length array.");
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
#if DEGREE_ORDER
		g.degree_ = computeGlobalDegree(g);
#endif
#if ISOLATE_FIRST_EDGE || DEGREE_ORDER
		isolateFirstEdge(g);
#endif // #if ISOLATE_FIRST_EDGE || DEGREE_ORDER
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
			int64_t local_bits_max = int64_t(1) << local_bits_;
			print_with_prefix("non zero rows. Total %f M / %f M = %f %% Avg %f M / %f M Max %f %%+",
					to_mega(sum_rowbmp[0]), to_mega(sum_rowbmp[1]), to_mega(sum_rowbmp[0]) / to_mega(sum_rowbmp[1]) * 100,
					to_mega(sum_rowbmp[0]) / mpi.size_2d, to_mega(sum_rowbmp[1]) / mpi.size_2d,
					diff_percent(max_rowbmp[0], send_rowbmp[0], mpi.size_2d));
			print_with_prefix("distributed edges. Total %f M Avg %f M Max %f %%+",
					to_mega(sum_rowbmp[2]), to_mega(sum_rowbmp[2]) / mpi.size_2d,
					diff_percent(max_rowbmp[2], sum_rowbmp[2], mpi.size_2d));
#if BFELL
			print_with_prefix("column_length size (opt/naive). Total %f M / %f M Avg %f M / %f M Max %f %%+ / %f %%+",
					to_mega(sum_rowbmp[3]), to_mega(sum_rowbmp[4]),
					to_mega(sum_rowbmp[3]) / mpi.size_2d, to_mega(sum_rowbmp[4]) / mpi.size_2d,
					diff_percent(max_rowbmp[3], sum_rowbmp[3], mpi.size_2d),
					diff_percent(max_rowbmp[4], sum_rowbmp[4], mpi.size_2d));
#endif
			print_with_prefix("Type requirements:");
			print_with_prefix("Global vertex id %s using %s", minimum_type(num_local_verts * mpi.size_2d), TypeName<int64_t>::value);
			print_with_prefix("Local vertex id %s using %s", minimum_type(num_local_verts), TypeName<uint32_t>::value);
			print_with_prefix("Index for local edges %s using %s", minimum_type(max_rowbmp[2]), TypeName<int64_t>::value);
			print_with_prefix("*Index for src local region %s using %s", minimum_type(local_bits_max * mpi.size_2dc), TypeName<TwodVertex>::value);
			print_with_prefix("*Index for dst local region %s using %s", minimum_type(local_bits_max * mpi.size_2dr), TypeName<TwodVertex>::value);
			print_with_prefix("Index for non zero rows %s using %s", minimum_type(max_rowbmp[0]), TypeName<TwodVertex>::value);
			print_with_prefix("*BFELL sort region size %s using %s", minimum_type(BFELL_SORT), TypeName<SortIdx>::value);
			print_with_prefix("Memory consumption:");
			print_with_prefix("row_bitmap %f MB", to_mega(row_bitmap_length*sizeof(BitmapType)));
			print_with_prefix("row_sums %f MB", to_mega((row_bitmap_length+1)*sizeof(TwodVertex)));
			print_with_prefix("edge_array %f MB", to_mega(max_rowbmp[2]*sizeof(TwodVertex)));
#if BFELL
			print_with_prefix("block_offset %f MB", to_mega((src_region_length/BFELL_SORT)*sizeof(TwodVertex)*2));
			print_with_prefix("sorted_idx %f MB", to_mega(max_rowbmp[0]*sizeof(SortIdx)));
			print_with_prefix("column_length(opt) %f MB", to_mega(max_rowbmp[3]*sizeof(SortIdx)));
			print_with_prefix("column_length(naive) %f MB", to_mega(max_rowbmp[4]*sizeof(SortIdx)));
#else
			print_with_prefix("row_starts %f MB", to_mega(row_bitmap_length*sizeof(BitmapType)));
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
		const int local_bits = local_bits_;
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			const SeparatedId v0_src(vertex_owner_c(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_src(vertex_owner_c(v1), vertex_local(v1), local_bits);
			const SeparatedId v0_dst(vertex_owner_r(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_dst(vertex_owner_r(v1), vertex_local(v1), local_bits);
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_src.value, v1_dst.value, edge_data[i].weight_);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_src.value, v0_dst.value, edge_data[i].weight_);
		} // #pragma omp for schedule(static)
	}

	// function #2
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::no_weight dummy = 0)
	{
		const int local_bits = local_bits_;
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			const SeparatedId v0_src(vertex_owner_c(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_src(vertex_owner_c(v1), vertex_local(v1), local_bits);
			const SeparatedId v0_dst(vertex_owner_r(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_dst(vertex_owner_r(v1), vertex_local(v1), local_bits);
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_src.value, v1_dst.value);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_src.value, v0_dst.value);
		} // #pragma omp for schedule(static)
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		const int64_t L = g.num_local_verts_;
		const int lgl = local_bits_;
		const int log_local_src = local_bits_ + get_msb_index(mpi.size_2dc-1) + 1;
#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const SeparatedId v0(edges[i].v0());
			const SeparatedId v1(edges[i].v1());
			const int weight = edges[i].weight_;

			const int src_high = v0.compact(lgl, L) >> LOG_EDGE_PART_SIZE;
			const uint16_t src_low = v0.compact(lgl, L) & EDGE_PART_SIZE_MASK;
			const int64_t pos = __sync_fetch_and_add(&wide_row_starts_[src_high], 1);

			// random access (write)
#ifndef NDEBUG
			assert( g.edge_array_[pos] == 0 );
#endif
			src_vertexes_[pos] = src_low;
			g.edge_array_[pos] = (weight << log_local_src) | v1.value;
		}
	}

	// function #2
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
		const int64_t L = g.num_local_verts_;
		const int lgl = local_bits_;
#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const SeparatedId v0(edges[i].v0());
			const SeparatedId v1(edges[i].v1());

			const int src_high = v0.compact(lgl, L) >> LOG_EDGE_PART_SIZE;
			const uint16_t src_low = v0.compact(lgl, L) & EDGE_PART_SIZE_MASK;
			const int64_t pos = __sync_fetch_and_add(&wide_row_starts_[src_high], 1);

			// random access (write)
#ifndef NDEBUG
			assert( g.edge_array_[pos] == 0 );
#endif
			src_vertexes_[pos] = src_low;
			g.edge_array_[pos] = v1.value;
		}
	}

	void scatterAndStore(EdgeList* edge_list, GraphType& g) {
		TRACER(store_edge);
		ScatterContext scatter(mpi.comm_2d);
		EdgeType* edges_to_send = static_cast<EdgeType*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(EdgeType)));

		const int64_t num_local_verts = g.num_local_verts_;
		g.edge_array_ = (TwodVertex*)cache_aligned_xcalloc(wide_row_starts_[num_wide_rows_]*sizeof(TwodVertex));
		src_vertexes_ = (uint16_t*)cache_aligned_xcalloc(wide_row_starts_[num_wide_rows_]*sizeof(uint16_t));

		int num_loops = edge_list->beginRead(true);

		if(mpi.isMaster()) print_with_prefix("Begin construction. Number of iterations is %d.", num_loops);

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

			if(mpi.isMaster()) print_with_prefix("Scatter edges.");

			EdgeType* recv_edges = scatter.scatter(edges_to_send);
			const int num_recv_edges = scatter.get_recv_count();

			if(mpi.isMaster()) print_with_prefix("Add edges.");

			addEdges(recv_edges, num_recv_edges, g);

			scatter.free(recv_edges);

			if(mpi.isMaster()) print_with_prefix("Iteration %d finished.", loop_count);
		}

		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		if(mpi.isMaster()) print_with_prefix("Refreshing edge offset.");
		memmove(wide_row_starts_+1, wide_row_starts_, num_wide_rows_*sizeof(wide_row_starts_[0]));
		wide_row_starts_[0] = 0;

#ifndef NDEBUG
#pragma omp parallel for
		for(int64_t i = 0; i <= num_wide_rows_; ++i) {
			if(row_starts_sup_[i] != wide_row_starts_[i]) {
				print_with_prefix("Error: Edge Counts: i=%"PRId64",1st=%"PRId64",2nd=%"PRId64"", i, row_starts_sup_[i], wide_row_starts_[i]);
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
#pragma omp for
		for(int64_t i = 0; i < num_wide_rows_; ++i) {
			const int64_t edge_offset = wide_row_starts_[i];
			const int64_t edge_count = wide_row_starts_[i+1] - edge_offset;

			// sort
			sort2(src_vertexes_ + edge_offset, g.edge_array_ + edge_offset, edge_count, SortEdgeCompair());

			int64_t idx = edge_offset;
			if(edge_count > 0) {
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
			}
			row_starts_sup_[i] = idx - edge_offset;
		} // #pragma omp for

#undef ENCODE
#undef DECODE
	}

	void sortEdges(GraphType& g) {
		TRACER(sort_edge);
		if(mpi.isMaster()) print_with_prefix("Sorting edges.");

#pragma omp parallel
		sortEdgesInner<EdgeType>(g);

		// this loop can't be parallel
		int64_t rowstart_new = 0;
		for(int64_t i = 0; i < num_wide_rows_; ++i) {
			const int64_t edge_count_new = row_starts_sup_[i];
			const int64_t rowstart_old = wide_row_starts_[i]; // read before write
			wide_row_starts_[i] = rowstart_new;
			if(rowstart_new != rowstart_old) {
				memmove(src_vertexes_ + rowstart_new, src_vertexes_ + rowstart_old, edge_count_new * sizeof(src_vertexes_[0]));
				memmove(g.edge_array_ + rowstart_new, g.edge_array_ + rowstart_old, edge_count_new * sizeof(g.edge_array_[0]));
			}
			rowstart_new += edge_count_new;
		}
		const int64_t old_num_edges = wide_row_starts_[num_wide_rows_];
		wide_row_starts_[num_wide_rows_] = rowstart_new;

		 int64_t num_edge_sum[2] = {0};
		 int64_t num_edge[2] = {old_num_edges, rowstart_new};
		MPI_Reduce(num_edge, num_edge_sum, 2, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);
		if(mpi.isMaster()) print_with_prefix("# of edges is reduced. Total %zd -> %zd Diff %f %%",
				num_edge_sum[0], num_edge_sum[1], (double)(num_edge_sum[0] - num_edge_sum[1])/(double)num_edge_sum[0]*100.0);
		g.num_global_edges_ = num_edge_sum[1];
	}

	void computeNumVertices(GraphType& g) {
		TRACER(num_verts);
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t local_bitmap_width = num_local_verts / NBPE;
		int recvcounts[mpi.size_2dc];
		for(int i = 0; i < mpi.size_2dc; ++i) recvcounts[i] = local_bitmap_width;

		g.has_edge_bitmap_ = (BitmapType*)cache_aligned_xmalloc(local_bitmap_width*sizeof(BitmapType));
		MPI_Reduce_scatter(g.row_bitmap_, g.has_edge_bitmap_, recvcounts, MpiTypeOf<BitmapType>::type, MPI_BOR, mpi.comm_2dr);
		int64_t num_vertices = 0;
#pragma omp parallel for reduction(+:num_vertices)
		for(int i = 0; i < local_bitmap_width; ++i) {
			num_vertices += __builtin_popcountl(g.has_edge_bitmap_[i]);
		}
		int64_t tmp_send_num_vertices = num_vertices;
		MPI_Allreduce(&tmp_send_num_vertices, &num_vertices, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
		VERVOSE(int64_t num_virtual_vertices = int64_t(1) << g.log_actual_global_verts_);
		VERVOSE(if(mpi.isMaster()) print_with_prefix("# of actual vertices %f G %f %%", to_giga(num_vertices),
				(double)num_vertices / (double)num_virtual_vertices * 100.0));
		g.num_global_verts_ = num_vertices;
	}

	//const int log_size_;
	//const int rmask_;
	//const int cmask_;
	int log_local_verts_unit_;
	int64_t num_wide_rows_;

	int local_bits_;

	uint16_t* src_vertexes_;
	int64_t* wide_row_starts_;
	int64_t* row_starts_sup_;
};

} // namespace detail {


#endif /* GRAPH_CONSTRUCTOR_HPP_ */
