/*
 * bottom_up_comm.hpp
 *
 *  Created on: 2014/06/04
 *      Author: ueno
 */

#ifndef BOTTOM_UP_COMM_HPP_
#define BOTTOM_UP_COMM_HPP_

#include "parameters.h"
#include "abstract_comm.hpp"
#include "utils.hpp"

#define debug(...) debug_print(BUCOM, __VA_ARGS__)

struct BottomUpSubstepTag {
	int64_t length;
	int region_id; // < 1024
	int routed_count; // <= 1024
	int route; // <= 1
};

struct BottomUpSubstepData {
	BottomUpSubstepTag tag;
	void* data;
};

class BottomUpSubstepCommBase {
public:
	BottomUpSubstepCommBase() { }
	virtual ~BottomUpSubstepCommBase() {
#if OVERLAP_WAVE_AND_PRED
		MPI_Comm_free(&mpi_comm);
#endif
	}
	void init(MPI_Comm mpi_comm__) {
		mpi_comm = mpi_comm__;
#if OVERLAP_WAVE_AND_PRED
		MPI_Comm_dup(mpi_comm__, &mpi_comm);
#endif
		int size, rank;
		MPI_Comm_size(mpi_comm__, &size);
		MPI_Comm_rank(mpi_comm__, &rank);
		// compute route
		int right_rank = (rank + 1) % size;
		int left_rank = (rank + size - 1) % size;
		nodes(0).rank = left_rank;
		nodes(1).rank = right_rank;
		debug("left=%d, right=%d", left_rank, right_rank);
	}
	void send_first(BottomUpSubstepData* data) {
		data->tag.routed_count = 0;
		data->tag.route = send_filled % 2;
		debug("send_first length=%d, send_filled=%d", data->tag.length, send_filled);
		send_pair[send_filled++] = *data;
		if(send_filled == 2) {
			send_recv();
			send_filled = 0;
		}
	}
	void send(BottomUpSubstepData* data) {
		debug("send length=%d, send_filled=%d", data->tag.length, send_filled);
		send_pair[send_filled++] = *data;
		if(send_filled == 2) {
			send_recv();
			send_filled = 0;
		}
	}
	void recv(BottomUpSubstepData* data) {
		if(recv_tail >= recv_filled) {
			next_recv();
			if(recv_tail >= recv_filled) {
				fprintf(IMD_OUT, "recv_tail >= recv_filled\n");
				throw "recv_filled >= recv_tail";
			}
		}
		*data = recv_pair[recv_tail++ % NBUF];
		debug("recv length=%d, recv_tail=%d", data->tag.length, recv_tail - 1);
	}
	void finish() {
	}

	virtual void print_stt() {
	}
protected:
	enum {
		NBUF = 4,
		BUFMASK = NBUF-1,
	};

	struct CommTargetBase {
		int rank;
	};

	MPI_Comm mpi_comm;
	std::vector<void*> free_list;
	BottomUpSubstepData send_pair[NBUF];
	BottomUpSubstepData recv_pair[NBUF];

	int element_size;
	int buffer_width;
	int send_filled;
	int recv_filled;
	int recv_tail;

	virtual CommTargetBase& nodes(int target) = 0;
	virtual void send_recv() = 0;
	virtual void next_recv() = 0;

	int buffers_available() {
		return (int)free_list.size();
	}

	void* get_buffer() {
		assert(buffers_available());
		void* ptr = free_list.back();
		free_list.pop_back();
		return ptr;
	}

	void free_buffer(void* buffer) {
		free_list.push_back(buffer);
	}

	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__) {
		element_size = sizeof(T);
		buffer_width = buffer_width__;

		free_list.clear();
		for(int i = 0; i < buffer_count__; ++i) {
			free_list.push_back(recv_buffers__[i]);
		}
		send_filled = recv_tail = recv_filled = 0;

		debug("begin buffer_count=%d, buffer_width=%d",
				buffer_count__, buffer_width__);

	}
};

class MpiBottomUpSubstepComm : public BottomUpSubstepCommBase {
	typedef BottomUpSubstepCommBase super__;
public:
	MpiBottomUpSubstepComm(MPI_Comm mpi_comm__)
	{
		init(mpi_comm__);
	}
	virtual ~MpiBottomUpSubstepComm() {
	}
	void register_memory(void* memory, int64_t size) {
	}
	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__) {
		super__::begin(recv_buffers__, buffer_count__, buffer_width__);
		type = MpiTypeOf<T>::type;
		recv_top = 0;
		is_active = false;
	}
	void probe() {
		next_recv_probe(false);
	}
	void finish() {
	}

protected:
	struct CommTarget : public CommTargetBase {
	};

	CommTarget nodes_[2];
	MPI_Datatype type;
	MPI_Request req[4];
	int recv_top;
	bool is_active;

	virtual CommTargetBase& nodes(int target) { return nodes_[target]; }

	int make_tag(BottomUpSubstepTag& tag) {
		//return (1 << 30) | (tag.route << 24) |
		return (tag.route << 24) |
				(tag.routed_count << 12) | tag.region_id;
	}

	BottomUpSubstepTag make_tag(MPI_Status& status) {
		BottomUpSubstepTag tag;
		int length;
		int raw_tag = status.MPI_TAG;
		MPI_Get_count(&status, type, &length);
		tag.length = length;
		tag.region_id = raw_tag & 0xFFF;
		tag.routed_count = (raw_tag >> 12) & 0xFFF;
		tag.route = (raw_tag >> 24) & 1;
		return tag;
	}

	void next_recv_probe(bool blocking) {
		if(is_active) {
			MPI_Status status[4];
			if(blocking) {
				MPI_Waitall(4, req, status);
			}
			else {
				int flag;
				MPI_Testall(4, req, &flag, status);
				if(flag == false) {
					return ;
				}
			}
			int recv_0 = recv_filled++ % NBUF;
			int recv_1 = recv_filled++ % NBUF;
			recv_pair[recv_0].tag = make_tag(status[0]);
			recv_pair[recv_1].tag = make_tag(status[1]);
			free_buffer(send_pair[2].data);
			free_buffer(send_pair[3].data);
			is_active = false;
		}
	}

	virtual void next_recv() {
		next_recv_probe(true);
	}

	virtual void send_recv() {
		next_recv_probe(true);
		int recv_0 = recv_top++ % NBUF;
		int recv_1 = recv_top++ % NBUF;
		recv_pair[recv_0].data = get_buffer();
		recv_pair[recv_1].data = get_buffer();
		MPI_Irecv(recv_pair[recv_0].data, buffer_width,
				type, nodes_[0].rank, MPI_ANY_TAG, mpi_comm, &req[0]);
		MPI_Irecv(recv_pair[recv_1].data, buffer_width,
				type, nodes_[1].rank, MPI_ANY_TAG, mpi_comm, &req[1]);

		MPI_Isend(send_pair[0].data, send_pair[0].tag.length,
				type, nodes_[1].rank, make_tag(send_pair[0].tag), mpi_comm, &req[2]);
		MPI_Isend(send_pair[1].data, send_pair[1].tag.length,
				type, nodes_[0].rank, make_tag(send_pair[1].tag), mpi_comm, &req[3]);

		send_pair[2] = send_pair[0];
		send_pair[3] = send_pair[1];
		is_active = true;
#if !BOTTOM_UP_OVERLAP_PFS // if overlapping is disabled
		next_recv_probe(true);
#endif
	}
};

#undef debug

#endif /* BOTTOM_UP_COMM_HPP_ */
