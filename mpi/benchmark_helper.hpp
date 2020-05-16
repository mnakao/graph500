#ifndef BENCHMARK_HELPER_HPP_
#define BENCHMARK_HELPER_HPP_

#include "logfile.h"

class ProgressReport
{
public:
	ProgressReport(int max_progress)
		: max_progress_(max_progress)
		, my_progress_(0)
		, send_req_(new MPI_Request[max_progress]())
		, recv_req_(NULL)
		, send_buf_(new int[max_progress]())
		, recv_buf_(NULL)
		, g_progress_(NULL)
	{
		for(int i = 0; i < max_progress; ++i) {
			send_req_[i] = MPI_REQUEST_NULL;
			send_buf_[i] = i + 1;
		}
		pthread_mutex_init(&thread_sync_, NULL);
		if(mpi.isMaster()) {
			recv_req_ = new MPI_Request[mpi.size_2d]();
			recv_buf_  = new int[mpi.size_2d]();
			g_progress_ = new int[mpi.size_2d]();
			for(int i = 0; i < mpi.size_2d; ++i) {
				recv_req_[i] = MPI_REQUEST_NULL;
			}
		}
	}
	~ProgressReport() {
		pthread_mutex_destroy(&thread_sync_);
		delete [] send_req_; send_req_ = NULL;
		delete [] recv_req_; recv_req_ = NULL;
		delete [] send_buf_; send_buf_ = NULL;
		delete [] recv_buf_; recv_buf_ = NULL;
		delete [] g_progress_; g_progress_ = NULL;
	}
	void begin_progress() {
		my_progress_ = 0;
		if(mpi.isMaster()) {
			pthread_create(&thread_, NULL, update_status_thread, this);
			print_with_prefix("Begin Reporting Progress. Info: Rank is 2D.");
		}
	}
	void advace() {
		pthread_mutex_lock(&thread_sync_);
		MPI_Isend(&send_buf_[my_progress_], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_req_[my_progress_]);
		int index, flag;
		MPI_Testany(max_progress_, send_req_, &index, &flag, MPI_STATUS_IGNORE);
		pthread_mutex_unlock(&thread_sync_);
		++my_progress_;
	}
	void end_progress() {
		if(mpi.isMaster()) {
			pthread_join(thread_, NULL);
		}
		MPI_Waitall(max_progress_, send_req_, MPI_STATUSES_IGNORE);
	}

private:
	static void* update_status_thread(void* this_) {
		static_cast<ProgressReport*>(this_)->update_status();
		return NULL;
	}
	// return : complete or not
	void update_status() {
		for(int i = 0; i < mpi.size_2d; ++i) {
			g_progress_[i] = 0;
			recv_buf_[i] = 0; // ?????
			MPI_Irecv(&recv_buf_[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_req_[i]);
		}
		int* tmp_progress = new int[mpi.size_2d];
		int* node_list = new int[mpi.size_2d];
		bool complete = false;
		int work_count = 0;
		double print_time = MPI_Wtime();
		while(complete == false) {
			usleep(50*1000); // sleep 50 ms
			if(MPI_Wtime() - print_time >= 2.0) {
				print_time = MPI_Wtime();
				for(int i = 0; i < mpi.size_2d; ++i) {
					tmp_progress[i] = g_progress_[i];
					node_list[i] = i;
				}
				sort2(tmp_progress, node_list, mpi.size_2d);
				print_prefix();
				fprintf(IMD_OUT, "(Rank,Iter)=");
				for(int i = 0; i < std::min(mpi.size_2d, 8); ++i) {
					fprintf(IMD_OUT, "(%d,%d)", node_list[i], tmp_progress[i]);
				}
				fprintf(IMD_OUT, "\n");
			}
			pthread_mutex_lock(&thread_sync_);
			while(true) {
				int index, flag;
				MPI_Testany(mpi.size_2d, recv_req_, &index, &flag, MPI_STATUS_IGNORE);
				if(flag == 0) {
					if(++work_count > mpi.size_2d*2) {
						work_count = 0;
						break;
					}
					continue;
				}
				if(index == MPI_UNDEFINED) {
					complete = true;
					break;
				}
				g_progress_[index] = recv_buf_[index];
				if(g_progress_[index] < max_progress_) {
					MPI_Irecv(&recv_buf_[index], 1, MPI_INT, index, 0, MPI_COMM_WORLD, &recv_req_[index]);
				}
			}
			pthread_mutex_unlock(&thread_sync_);
		}
		delete [] tmp_progress;
		delete [] node_list;
	}

	pthread_t thread_;
	pthread_mutex_t thread_sync_;
	int max_progress_;
	int my_progress_;
	MPI_Request *send_req_; // length=max_progress
	MPI_Request *recv_req_; // length=mpi.size
	int* send_buf_; // length=max_progress
	int* recv_buf_; // length=mpi.size
	int* g_progress_; // length=mpi.size
};

template <typename EdgeList>
void generate_graph(EdgeList* edge_list, const GraphGenerator<typename EdgeList::edge_type>* generator)
{
	typedef typename EdgeList::edge_type EdgeType;
	EdgeType* edge_buffer = static_cast<EdgeType*>
						(cache_aligned_xmalloc(EdgeList::CHUNK_SIZE*sizeof(EdgeType)));
	edge_list->beginWrite();
	const int64_t num_global_edges = generator->num_global_edges();
	const int64_t num_global_chunks = (num_global_edges + EdgeList::CHUNK_SIZE - 1) / EdgeList::CHUNK_SIZE;
	const int64_t num_iterations = (num_global_chunks + mpi.size_2d - 1) / mpi.size_2d;
	double logging_time = MPI_Wtime();
#if REPORT_GEN_RPGRESS
	ProgressReport* report = new ProgressReport(num_iterations);
#endif
	if(mpi.isMaster()) {
		double global_data_size = (double)num_global_edges * 16.0 / 1000000000.0;
		double local_data_size = global_data_size / mpi.size_2d;
		print_with_prefix("Graph data size: %f GB ( %f GB per process )", global_data_size, local_data_size);
		print_with_prefix("Using storage: %s", edge_list->data_is_in_file() ? "yes" : "no");
		if(edge_list->data_is_in_file()) {
			print_with_prefix("Filepath: %s 1 2 ...", edge_list->get_filepath());
		}
		print_with_prefix("Communication chunk size: %d", EdgeList::CHUNK_SIZE);
		print_with_prefix("Generating graph: Total number of iterations: %" PRId64 "", num_iterations);
	}
#if REPORT_GEN_RPGRESS
	report->begin_progress();
#endif
#pragma omp parallel
	for(int64_t i = 0; i < num_iterations; ++i) {
		SET_OMP_AFFINITY;
		const int64_t start_edge = std::min((mpi.size_2d*i + mpi.rank_2d) * EdgeList::CHUNK_SIZE, num_global_edges);
		const int64_t end_edge = std::min(start_edge + EdgeList::CHUNK_SIZE, num_global_edges);
		generator->generateRange(edge_buffer, start_edge, end_edge);
#if defined(__INTEL_COMPILER)
#pragma omp barrier
#endif
		// we need to synchronize before this code.
		// There is the implicit barrier on the end of for loops.
#pragma omp master
		{
			edge_list->write(edge_buffer, end_edge - start_edge);

			if(mpi.isMaster()) {
				print_with_prefix("Time for iteration %" PRId64 " is %f ", i, MPI_Wtime() - logging_time);
				logging_time = MPI_Wtime();
			}
#if REPORT_GEN_RPGRESS
			report->advace();
#endif
		}
#pragma omp barrier

	}
#if REPORT_GEN_RPGRESS
	report->end_progress();
	delete report; report = NULL;
#endif
	edge_list->endWrite();
	free(edge_buffer);
	if(mpi.isMaster()) print_with_prefix("Finished generating.");
}

template <typename EdgeList>
void generate_graph_spec2010(EdgeList* edge_list, int scale, int edge_factor, int max_weight = 0)
{
	RmatGraphGenerator<typename EdgeList::edge_type, 5700, 1900> generator(scale, edge_factor, 255,
			PRM::USERSEED1, PRM::USERSEED2, InitialEdgeType::NONE);
	generate_graph(edge_list, &generator);
}

int read_log_file(LogFileFormat* log, int SCALE, int edgefactor, double* bfs_times, double* validate_times, double* edge_counts)
{
	int resume_root_idx = 0;
	const char* logfilename = getenv("LOGFILE");
	if(logfilename) {
		if(mpi.isMaster()) {
			FILE* fp = fopen(logfilename, "rb");
			if(fp != NULL) {
				fread(log, sizeof(log[0]), 1, fp);
				if(log->scale != SCALE || log->edge_factor != edgefactor || log->mpi_size != mpi.size_2d) {
					print_with_prefix("Log file is not match the current run: params:(current),(log): SCALE:%d,%d, edgefactor:%d,%d, size:%d,%d",
					SCALE, log->scale, edgefactor, log->edge_factor, mpi.size_2d, log->mpi_size);
					resume_root_idx = -2;
				}
				else {
					resume_root_idx = log->num_runs;
					fprintf(IMD_OUT, "===== LOG START =====\n");
					fprintf(IMD_OUT, "graph_generation:               %f s\n", log->generation_time);
					fprintf(IMD_OUT, "construction_time:              %f s\n", log->construction_time);
					int i;
					for (i = 0; i < resume_root_idx; ++i) {
						fprintf(IMD_OUT, "Running BFS %d\n", i);
						fprintf(IMD_OUT, "Time for BFS %d is %f\n", i, log->times[i].bfs_time);
						fprintf(IMD_OUT, "Validating BFS %d\n", i);
						fprintf(IMD_OUT, "Validate time for BFS %d is %f\n", i, log->times[i].validate_time);
						fprintf(IMD_OUT, "TEPS for BFS %d is %g\n", i, log->times[i].edge_counts / log->times[i].bfs_time);

						bfs_times[i] = log->times[i].bfs_time;
						validate_times[i] = log->times[i].validate_time;
						edge_counts[i] = log->times[i].edge_counts;
					}
					fprintf(IMD_OUT, "=====  LOG END  =====\n");

				}
				fclose(fp);
			}
		}
		MPI_Bcast(&resume_root_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if(resume_root_idx == -2) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	return resume_root_idx;
}

#endif /* BENCHMARK_HELPER_HPP_ */
