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

class BottomUpSubstepCommBase : public AsyncCommHandler {
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
		int left_rank = (rank + 1) % size;
		int right_rank = (rank + size - 1) % size;
		nodes(0).rank = left_rank;
		nodes(1).rank = right_rank;
		debug("left=%d, right=%d", left_rank, right_rank);
	}
	void send_first(BottomUpSubstepData* data) {
		send_queue[send_filled] = *data;
		send_queue[send_filled].tag.routed_count = 0;
		send_queue[send_filled].tag.route = send_filled % 2;
		debug("send_first length=%d, send_filled=%d", data->tag.length, send_filled);
		__sync_synchronize();
		++send_filled;
		__sync_synchronize();
	}
	void send(BottomUpSubstepData* data) {
		send_queue[send_filled] = *data;
		debug("send length=%d, send_filled=%d", data->tag.length, send_filled);
		__sync_synchronize();
		++send_filled;
		__sync_synchronize();
	}
	void recv(BottomUpSubstepData* data) {
		while(recv_filled <= recv_tail) ;
		__sync_synchronize();
		*data = recv_queue[recv_tail];
		debug("recv length=%d, recv_tail=%d", data->tag.length, recv_tail);
		++recv_tail;
	}
	void finish() {
		VT_TRACER("bu_comm_fin_wait");
		while(!finished) sched_yield();
		debug("user finished");
	}
	virtual void probe(void* comm_data) {
		if(finished) return;

		if(initialized == false) {
			begin_comm();
			initialized = true;
		}

		// pump send data
		while(send_tail < send_filled) {
			__sync_synchronize();
			BottomUpSubstepData comm_buf = send_queue[send_tail];
			int route = comm_buf.tag.route;
			assert(route == 0 || route == 1);
			nodes(route).send_queue.push_back(comm_buf);
			++send_tail;
		}

		probe_comm(comm_data);

		// finish ?
		int total_send = nodes(0).send_complete_count + nodes(1).send_complete_count;
		int total_recv = nodes(0).recv_complete_count + nodes(1).recv_complete_count;
		if(total_steps == total_send && total_steps == total_recv) {
			end_comm(comm_data);
			debug("finished");
			finished = true;
		}
	}

protected:
	enum {
		NBUF = 4,
		BUFMASK = NBUF-1,
	};

	struct CommTargetBase {
		int rank;
		std::deque<BottomUpSubstepData> send_queue;

		BottomUpSubstepData send_buf[NBUF];
		BottomUpSubstepData recv_buf[NBUF];

		unsigned int send_count; // the next buffer index is calculated from this value
		unsigned int send_complete_count;
		unsigned int recv_count; //   "
		unsigned int recv_complete_count;

		CommTargetBase() {
			for(int i = 0; i < NBUF; ++i) {
				send_buf[i].data = NULL;
				recv_buf[i].data = NULL;
			}
			send_count = recv_count = recv_complete_count = 0;
		}

		void clear_state() {
			send_count = send_complete_count = recv_count = recv_complete_count = 0;
			for(int i = 0; i < NBUF; ++i) {
				send_buf[i].data = NULL;
				recv_buf[i].data = NULL;
			}
		}
	};

	MPI_Comm mpi_comm;
	std::vector<void*> free_list;
	std::vector<BottomUpSubstepData> send_queue;
	std::vector<BottomUpSubstepData> recv_queue;

	int element_size;
	int total_steps;
	int buffer_width;
	bool initialized;
	volatile int send_filled;
	int send_tail; // only comm thread modify
	volatile int recv_filled;
	int recv_tail; // only user thred modify
	volatile bool finished;

	virtual CommTargetBase& nodes(int target) = 0;
	virtual void begin_comm() = 0;
	virtual void probe_comm(void* comm_data) = 0;
	virtual void end_comm(void* comm_data) = 0;

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

	void recv_data(BottomUpSubstepData* data) {
		// increment counter
		data->tag.routed_count++;
		recv_queue[recv_filled] = *data;
		__sync_synchronize();
		++recv_filled;
	}

	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__, int total_steps__) {
		element_size = sizeof(T);
		buffer_width = buffer_width__;

		free_list.clear();
		for(int i = 0; i < buffer_count__; ++i) {
			free_list.push_back(recv_buffers__[i]);
		}
		total_steps = total_steps__;
		send_queue.resize(total_steps, BottomUpSubstepData());
		recv_queue.resize(total_steps, BottomUpSubstepData());
		send_filled = send_tail = 0;
		recv_filled = recv_tail = 0;

		nodes(0).clear_state();
		nodes(1).clear_state();
		finished = false;
		initialized = false;
		debug("begin buffer_count=%d, buffer_width=%d, total_steps=%d",
				buffer_count__, buffer_width__, total_steps__);
	}
};

class MpiBottomUpSubstepComm : public BottomUpSubstepCommBase {
	typedef BottomUpSubstepCommBase super__;
public:
	MpiBottomUpSubstepComm(MPI_Comm mpi_comm__)
	{
		init(mpi_comm__);
		for(int i = 0; i < int(sizeof(req)/sizeof(req[0])); ++i) {
			req[i] = MPI_REQUEST_NULL;
		}
	}
	virtual ~MpiBottomUpSubstepComm() {
	}
	void register_memory(void* memory, int64_t size) {
	}
	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__, int total_steps__) {
		super__::begin(recv_buffers__, buffer_count__, buffer_width__, total_steps__);
		type = MpiTypeOf<T>::type;
	}
	virtual void begin_comm() {
		int send_per_node = total_steps / 2;
		int n = std::min<int>(NBUF, send_per_node);
		for(int i = 0; i < n && buffers_available(); ++i) {
			for(int p = 0; p < 2 && buffers_available(); ++p) {
				CommTarget& node = nodes_[p];
				void* buffer = get_buffer();
				node.recv_buf[i].data = buffer;
				MPI_Irecv(buffer, buffer_width, type, node.rank, MPI_ANY_TAG, mpi_comm, recv_request(p, i));
				debug("MPI_Irecv %srank=%d, buf_idx=%d", p ? ">" : "<", node.rank, i);
				node.recv_count++;
			}
		}
		debug("initialized");
	}
	virtual void probe_comm(void* comm_data) {
		int index; int flag; MPI_Status status = {0};
		MPI_Testany(sizeof(req)/sizeof(req[0]), req, &index, &flag, &status);
		if(flag != 0 && index != MPI_UNDEFINED) {
			int target; int buf_idx; bool b_send;
			req_info(index, target, buf_idx, b_send);

			CommTarget& node = nodes_[target];
			if(b_send) {
				free_buffer(node.send_buf[buf_idx].data);
				node.send_buf[buf_idx].data = NULL;
				node.send_complete_count++;
			}
			else {
				BottomUpSubstepTag tag = make_tag(status);
				node.recv_buf[buf_idx].tag = tag;
				debug("recv %srank=%d, buf_idx=%d, length=%d, region_id=%d, routed=%d",
						target ? ">" : "<", node.rank, buf_idx, tag.length, tag.region_id, tag.routed_count);
				recv_data(&node.recv_buf[buf_idx]);
				node.recv_buf[buf_idx].data = NULL;
				node.recv_complete_count++;
			}
		}

		// process receive completion
		for(int p = 0; p < 2; ++p) {
			set_recv_buffer(p);
			set_send_buffer(p);
		}

	}
	virtual void end_comm(void* comm_data) {
		//
	}

protected:
	struct CommTarget : public CommTargetBase {
	};

	CommTarget nodes_[2];
	MPI_Datatype type;
	MPI_Request req[NBUF*4];

	virtual CommTargetBase& nodes(int target) { return nodes_[target]; }

	MPI_Request* send_request(int target, int buf_idx) {
		return req + (target * NBUF + buf_idx) * 2;
	}

	MPI_Request* recv_request(int target, int buf_idx) {
		return send_request(target, buf_idx) + 1;
	}

	int make_tag(BottomUpSubstepTag& tag) {
		return (1 << 30) | (tag.route << 24) |
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

	void req_info(int index, int& target, int& buf_idx, bool& send) {
		send = ((index % 2) == 0);
		buf_idx = (index / 2) & BUFMASK;
		target = (index / 2 / NBUF);
	}

	void set_send_buffer(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(node.send_queue.size() > 0) {
			int buf_idx = node.send_count % NBUF;
			BottomUpSubstepData& comm_buf = node.send_buf[buf_idx];
			if(comm_buf.data != NULL) {
				break;
			}
			comm_buf = node.send_queue.front();
			node.send_queue.pop_front();
			int tag = make_tag(comm_buf.tag);
			MPI_Isend(comm_buf.data, comm_buf.tag.length, type, node.rank, tag,
					mpi_comm, send_request(target, buf_idx));
			debug("MPI_Isend %srank=%d, buf_idx=%d, length=%d, region_id=%d, routed_c=%d",
					target ? ">" : "<", node.rank, buf_idx, comm_buf.tag.length,
							comm_buf.tag.region_id, comm_buf.tag.routed_count);
#if VERVOSE_MODE
			if(element_size == 8) {
				g_bu_bitmap_comm += comm_buf.tag.length*element_size;
			}
			else {
				g_bu_list_comm += comm_buf.tag.length*element_size;
			}
#endif
			// increment counter
			node.send_count++;
		}
	}

	void set_recv_buffer(int target) {
		MY_TRACE;
		int send_per_node = total_steps / 2;
		CommTarget& node = nodes_[target];
		while(true) {
			if((int)node.recv_count >= send_per_node) {
				break;
			}
			int buf_idx = node.recv_count % NBUF;
			BottomUpSubstepData& comm_buf = node.recv_buf[buf_idx];
			if(comm_buf.data != NULL) {
				break;
			}
			if(buffers_available() == 0) {
				// no buffer
				break;
			}
			void* buffer = get_buffer();
			node.recv_buf[buf_idx].data = buffer;
			MPI_Irecv(buffer, buffer_width, type, node.rank,
					MPI_ANY_TAG, mpi_comm, recv_request(target, buf_idx));
			debug("MPI_Irecv r%sank=%d, buf_idx=%d", target ? ">" : "<", node.rank, buf_idx);
			// increment counter
			node.recv_count++;
		}
	}
};

//#if ENABLE_FJMPI_RDMA
#if 0
#include "fjmpi_comm.hpp"

class FJMpiBottomUpSubstepComm : public BottomUpSubstepCommBase {
public:
	FJMpiBottomUpSubstepComm(MPI_Comm mpi_comm__, int Z2, int rank_z1) {
		init(mpi_comm__, Z2, rank_z1);
		MPI_Group world_group;
		MPI_Group comm_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		MPI_Comm_group(mpi_comm__, &comm_group);
		int ranks1[] = { left_rank, right_rank };
		int ranks2[2];
		MPI_Group_translate_ranks(comm_group, 2, ranks1, world_group, ranks2);
		MPI_Group_free(&world_group);
		MPI_Group_free(&comm_group);

		nodes[0].pid = ranks2[0];
		nodes[0].put_flag = FJMPI_Local_nic[0] | FJMPI_Remote_nic[2] |
				FJMPI_RDMA_IMMEDIATE_RETURN | FJMPI_RDMA_PATH0;
		nodes[1].pid = ranks2[1];
		nodes[1].put_flag = FJMPI_Local_nic[1] | FJMPI_Remote_nic[3] |
				FJMPI_RDMA_IMMEDIATE_RETURN | FJMPI_RDMA_PATH0;

		current_step = 0;
	}
	virtual ~FJMpiBottomUpSubstepComm() {
	}
	void register_memory(void* memory, int64_t size) {
		rdma_buffer_pointers[0] = buffer_state;
		rdma_buffer_pointers[1] = memory;
		local_address[0] = FJMPI_Rdma_reg_mem(SYS_MEM_ID, buffer_state, sizeof(buffer_state));
		local_address[1] = FJMPI_Rdma_reg_mem(DATA_MEM_ID, memory, size);
		for(int i = 0; i < TNBUF*4; ++i) {
			buffer_state[i].state = INVALIDATED;
		}
		MPI_Barrier(mpi_comm);
		for(int i = 0; i < 2; ++i) {
			nodes[i].address[0] = FJMPI_Rdma_get_remote_addr(node[i].pid, SYS_MEM_ID);
			nodes[i].address[1] = FJMPI_Rdma_get_remote_addr(node[i].pid, DATA_MEM_ID);
		}
	}
	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__, int total_steps__) {
		begin(recv_buffers__, buffer_count__, buffer_width__, total_steps__);
		++current_step;
		debug("begin");
	}
	virtual void begin_comm() {
		debug("initialized");
	}
	virtual void probe_comm(void* comm_data) {
		// process receive completion
		for(int p = 0; p < 2; ++p) {
			check_recv_completion(p);
			set_recv_buffer(p);
			set_send_buffer(p);
		}

		// process completion
		std::vector<FJMPI_CQ>& cqs = *(std::vector<FJMPI_CQ>*)comm_data;
		for(int i = 0; i < cqs.size(); ++i) {
			FJMPI_CQ cq = cqs[i];
			if(cq.tag >= FIRST_USER_TAG && cq.tag < USER_TAG_END){
				int buf_idx = cq.tag - FIRST_USER_TAG;
				CommTarget& node = (cq.pid == nodes[0].pid) ? nodes[0] : nodes[1];
				BottomUpSubstepData& comm_buf = node.send_buf[buf_idx];
				free_list.push_back(comm_buf.data);
				debug("send complete to=%d, buf_idx=%d, length=%d", node.pid, buf_idx, comm_buf.tag.length);
				comm_buf.data = NULL;
				node.send_complete_count++;
			}
		}
	}
	virtual void end_comm(void* comm_data) {
		//
	}

protected:
	enum {
		FIRST_USER_TAG = 5,
		USER_TAG_END = FIRST_USER_TAG + NBUF,

		SYSTEM = 0,
		DATA = 1,

		SYS_MEM_ID = 300,
		DATA_MEM_ID = 301,

		INVALIDATED = 0,
		READY = 1,
		COMPLETE = 2,
	};

	struct BufferState {
		uint16_t state; // current state of the buffer
		uint16_t step;
		union {
			uint64_t offset; // offset to the buffer starting address
			BottomUpSubstepTag tag; // length of the received data in bytes
		};
	};

	struct CommTarget : public CommTargetBase {
		int pid;
		int put_flag;
		uint64_t address[2];

		CommTarget() : CommTargetBase() {
			pid = put_flag = 0;
			address[0] = 0;
			address[1] = 0;
		}

		uint64_t remote_address(int memory_id, uint64_t offset) {
			return address[memory_id] + offset;
		}
	};
	volatile BufferState buffer_state[NBUF*4];
	void* rdma_buffer_pointers[2];
	uint64_t local_address[2];
	CommTarget nodes_[2];
	int current_step;

	virtual CommTargetBase& nodes(int target) { return nodes_[target]; }

	template <typename T>
	uint64_t offset_from_pointer(T* pionter, int memory_id) const {
		return ((const uint8_t*)pionter - (const uint8_t*)rdma_buffer_pointers[memory_id]);
	}

	template <typename T>
	uint64_t local_address_from_pointer(T* pionter, int memory_id) const {
		return local_address[memory_id] +
				offset_from_pointer(pionter, memory_id);
	}

	volatile BufferState& send_buffer_state(int rank, int idx) {
		return buffer_state[offset_of_send_buffer_state(rank, idx)];
	}

	volatile BufferState& recv_buffer_state(int rank, int idx) {
		return buffer_state[offset_of_recv_buffer_state(rank, idx)];
	}

	static int offset_of_send_buffer_state(int rank, int idx) {
		return rank * NBUF * 2 + idx;
	}

	static int offset_of_recv_buffer_state(int rank, int idx) {
		return rank * NBUF * 2 + NBUF + idx;
	}

	void set_send_buffer(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(node.send_queue.size() > 0) {
			int buf_idx = node.send_count % NBUF;
			BottomUpSubstepData& comm_buf = node.send_buf[buf_idx];
			volatile BufferState& bs = send_buffer_state(target, buf_idx);
			if(comm_buf.data != NULL || bs.state != READY || bs.step != current_step) {
				debug("not ready to=%d, idx=%d(%d), state=%d", nodes[target].pid,
						node.send_count, buf_idx, send_buffer_state(target, buf_idx).state);
				break;
			}
			// To force loading state before loading other information
			__sync_synchronize();
			comm_buf = node.send_queue.front();
			node.send_queue.pop_front();
			volatile BufferState& buf_state = send_buffer_state(target, buf_idx);
			debug("set_send_buffer to=%d, idx=%d(%d), length=%d",
					nodes_[target].pid, node.send_count, buf_idx, comm_buf.tag.length);
			int pid = node.pid;
			{
				// input RDMA command to send data
				int memory_id = DATA_MEM_ID;
				int tag = FIRST_USER_TAG + buf_idx;
				uint64_t raddr = node.remote_address(DATA_MEM_ID, buf_state.offset);
				uint64_t laddr = local_address_from_pointer(comm_buf.data, DATA_MEM_ID);
				int64_t length = element_size * comm_buf.tag.length;
				FJMPI_Rdma_put(pid, tag, raddr, laddr, length, node.put_flag);
			}
			{
				// input RDMA command to notify completion
				// make buffer state for the statement of completion
				buf_state.state = COMPLETE;
				buf_state.step = current_step;
				buf_state.tag = comm_buf.tag;
				int tag = 0;
				uint64_t raddr = node.remote_address(SYS_MEM_ID,
						sizeof(BufferState) * offset_of_recv_buffer_state(target, buf_idx));
				uint64_t laddr = local_address_from_pointer(&buf_state, 0);
				FJMPI_Rdma_put(pid, tag, raddr, laddr, sizeof(BufferState), node.put_flag);
			}
			// increment counter
			node.send_count++;
		}
	}

	void set_recv_buffer(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(true) {
			int buf_idx = node.recv_count % NBUF;
			BottomUpSubstepData& comm_buf = node.recv_buf[buf_idx];
			if(comm_buf.data != NULL) {
				break;
			}
			assert (recv_buffer_state(target, buf_idx).state != READY ||
					recv_buffer_state(target, buf_idx).step != current_step);
			if(free_list.size() == 0) {
				// no buffer
				break;
			}
			// set new receive buffer
			comm_buf.data = free_list.back(); free_list.pop_back();
			int memory_id = DATA_MEM_ID;
			volatile BufferState& buf_state = recv_buffer_state(target, buf_idx);
			buf_state.state = READY;
			buf_state.step = current_step;
			buf_state.offset = offset_from_pointer(comm_buf.data, memory_id);
			// notify buffer info to the remote process
			int pid = node.pid;
			int tag = 0;
			uint64_t raddr = node.remote_address(SYS_MEM_ID,
					sizeof(BufferState) * offset_of_send_buffer_state(target, buf_idx));
			uint64_t laddr = local_address_from_pointer(&buf_state, SYS_MEM_ID);
			debug("set_recv_buffer to=%d, idx=%d(%d), state_address=0x%x",
					pid, node.recv_count, buf_idx, raddr);
			FJMPI_Rdma_put(pid, tag, raddr, laddr, sizeof(BufferState), node.put_flag);
			// increment counter
			node.recv_count++;
		}
	}

	void check_recv_completion(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(true) {
			int buf_idx = node.recv_complete_count % NBUF;
			BottomUpSubstepData& comm_buf = node.recv_buf[buf_idx];
			volatile BufferState& buf_state = recv_buffer_state(target, buf_idx);
			if(comm_buf.data == NULL || buf_state.state != COMPLETE) {
				break;
			}
			// To force loading state before loading length
			__sync_synchronize();

			// receive completed
			comm_buf.tag = buf_state.tag;
			debug("recv complete rank=%d, idx=%d(%d), length=%d",
					node.pid, node.recv_complete_count, buf_idx, comm_buf.tag.length);
			recv_data(&comm_buf);
			comm_buf.data = NULL;
			// increment counter
			node.recv_complete_count++;
		}
	}

};
#endif // #ifdef ENABLE_FJMPI_RDMA
#undef debug

#endif /* BOTTOM_UP_COMM_HPP_ */
