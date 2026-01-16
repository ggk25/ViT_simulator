import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.chip import chip ,chip_dict
import math
import itertools

class matmul:
    def __init__(self, data_type:DataType):
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.data_type = data_type
    
    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )
    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M ,self.N, self.K , self.data_type)
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)

        return output
    
    @staticmethod
    def simulate_multicore_mapping(M, N, K, B1, B2, NUM_PES=4, is_dynamic=False):
        """
        M, N, K: 全局矩阵大小
        B1: Shared SRAM 带宽 (elements/cycle/bank_group) - 假设广播只占1个带宽
        B2: Distributed RRAM 带宽 (elements/cycle/pe) - 每个PE独享
        """
        NUM_PES = 4
        
        # 候选硬件参数 (Tm, Tn, Tk)
        candidates = [
            (32, 32, 72),
            (16, 16, 144),
            (8, 8, 288),
            (4, 4, 576),
            (2, 2, 1152)
        ]
        
        results = []

        for Tm, Tn, Tk in candidates:
            # 硬件参数 (单个PE的Tile)
            size_tile_a = Tm * Tk
            size_tile_b = Tk * Tn
            size_tile_c = Tm * Tn * 4 #input是1byte，output是4byte
            compute_cycles = Tm
            
            # 策略对比：我们需要对比 Spatial 切分 M 还是 切分 N
            # Strategy 1: Split N (每个PE负责 N_global / 4) -> 预期最佳
            # Strategy 2: Split M (每个PE负责 M_global / 4) -> 预期较差
            strategies = [
                {'name': 'Spatial Split N (Broadcast A)', 'split_dim': 'n'}
            ]

            for strat in strategies:
                split_dim = strat['split_dim']
                
                # 根据切分策略分配每个PE的任务量
                pe_M = M // NUM_PES if split_dim == 'm' else M
                pe_N = N // NUM_PES if split_dim == 'n' else N
                pe_K = K # K一般不做空间切分，因为涉及到跨核累加(All-Reduce)，此处不考虑
                
                # 计算每个PE需要的 Steps
                steps_m = math.ceil(pe_M / Tm)
                steps_n = math.ceil(pe_N / Tn)
                steps_k = math.ceil(pe_K / Tk)
                dim_steps = {'m': steps_m, 'n': steps_n, 'k': steps_k}
                
                # 遍历 Loop Order (时间映射)
                for order in itertools.permutations(['m', 'n', 'k']):
                    total_cycles = 0
                    total_sram_cycles = 0
                    total_rram_cycles = 0
                    total_compute_cycles = 0
                    
                    # 状态追踪 (每个PE独立追踪自己的Tile变化)
                    # 结构: [PE0_State, PE1_State, ..., PE3_State]
                    prev_a_indices = [None] * NUM_PES
                    prev_b_indices = [None] * NUM_PES
                    prev_d_indices = [None] * NUM_PES
                    
                    outer, middle, inner = order
                    
                    # 模拟时间循环
                    for idx_outer in range(dim_steps[outer]):
                        for idx_middle in range(dim_steps[middle]):
                            for idx_inner in range(dim_steps[inner]):
                                
                                # --- 这一步是 4个PE 并行执行的 ---
                                
                                # 1. 收集所有PE在当前时刻对 SRAM (A, C/D) 和 RRAM/SRAM (B) 的请求
                                load_a_addr = [] # load A 的逻辑块坐标
                                load_b_addr = [] # 如果是 dynamic，则 load B 也去 SRAM
                                load_c_addr = [] # load C 的逻辑块坐标
                                store_d_addr = [] # store D 的逻辑块坐标
                                load_b_occurred = False # 只有在 !is_dynamic 时使用 RRAM 开销

                                idx_map = {
                                    outer: idx_outer,
                                    middle: idx_middle,
                                    inner: idx_inner
                                }
                                for pe_id in range(NUM_PES):
                                    # 计算当前PE负责的逻辑坐标偏移
                                    offset_m = (pe_id * steps_m) if split_dim == 'm' else 0
                                    offset_n = (pe_id * steps_n) if split_dim == 'n' else 0
                                    
                                    # 全局逻辑坐标 (用于判断地址是否相同)
                                    global_m_idx = offset_m + idx_map['m']
                                    global_n_idx = offset_n + idx_map['n']
                                    global_k_idx = idx_map['k']
                                    
                                    # --- 分析 A (SRAM) ---
                                    curr_a = (global_m_idx, global_k_idx)
                                    if curr_a != prev_a_indices[pe_id]:
                                        load_a_addr.append(curr_a) # 加入请求队列
                                        prev_a_indices[pe_id] = curr_a
                                    
                                    # --- 分析 B (RRAM/SRAM) ---
                                    curr_b = (global_k_idx, global_n_idx)
                                    if curr_b != prev_b_indices[pe_id]:
                                        if is_dynamic:
                                            load_b_addr.append(curr_b)
                                        else:
                                            # RRAM是独立的，每个PE自己读自己的，带宽不竞争
                                            load_b_occurred = True 
                                        prev_b_indices[pe_id] = curr_b
                                        
                                    # --- 分析 C/D (SRAM) ---
                                    curr_d = (global_m_idx, global_n_idx)
                                    if curr_d != prev_d_indices[pe_id]:
                                        # 切换Output Tile：需要 Store Old + Load New
                                        if prev_d_indices[pe_id] is not None:
                                            store_d_addr.append(prev_d_indices[pe_id]) # Store Request
                                        load_c_addr.append(curr_d) # Load Request
                                        prev_d_indices[pe_id] = curr_d

                                # --- 计算带宽开销 (关键逻辑) ---
                                
                                # 1. SRAM A 的开销：去重！
                                # 如果大家请求同一个地址，set长度为1 -> 广播
                                # 如果大家请求不同地址，set长度为4 -> 串行/竞争
                                unique_a_requests = set(load_a_addr)
                                cost_load_a = len(unique_a_requests) * size_tile_a
                                
                                # 2. SRAM C/D 的开销：去重 (虽然C通常很难广播，除了清零)
                                # 写回通常是冲突的，必须串行
                                unique_load_c_requests = set(load_c_addr)
                                cost_load_c = len(unique_load_c_requests) * size_tile_c

                                cost_store_d = len(store_d_addr) * size_tile_c

                                # 3. 分析 B 的位置 (RRAM vs SRAM)
                                if is_dynamic:
                                    unique_b_requests = set(load_b_addr)
                                    cost_load_b = len(unique_b_requests) * size_tile_b
                                    time_rram = 0
                                else:
                                    cost_load_b = 0
                                    # 因为是分布式的，且并行，只要发生了加载，耗时就是一个Tile的时间
                                    time_rram = math.ceil((size_tile_b / B2)) if load_b_occurred else 0
                                
                                # SRAM 读写耗时
                                time_sram_load = math.ceil((cost_load_a + cost_load_b + cost_load_c) / B1)
                                time_sram_store = math.ceil((cost_store_d) / B1)
                                
                                # 4. 计算核心耗时 (流水线瓶颈)
                                step_latency = max(time_sram_load, time_sram_store, time_rram, compute_cycles)
                                total_cycles += step_latency
                                total_sram_cycles += time_sram_load
                                total_rram_cycles += time_rram
                                total_compute_cycles += compute_cycles
                    
                    # 加上最后的Flush时间 (C写回)
                    # 假设写回不冲突情况下的理想带宽，或者算上冲突
                    # 这里简化处理：所有PE都要写回最后一块，互不相同
                    flush_cost = (NUM_PES * size_tile_c) / B1
                    total_cycles += flush_cost
                    total_sram_cycles += flush_cost

                    results.append({
                        "Strategy": strat['name'],
                        "Loop Order": "->".join(order) + " (Inner)",
                        "Cycles": int(total_cycles),
                        "Tm": Tm,
                        "Tn": Tn,
                        "Tk": Tk,
                        "Components": [int(total_sram_cycles), int(total_rram_cycles), int(total_compute_cycles), 0]
                    })

        results.sort(key=lambda x: x['Cycles'])
        return results[0]

    def mapping_and_simulate(self, chip: chip, is_MHA: bool):
        NUM_PES = chip.n_pe
        B1 = chip.sram_bandwidth_per_cycle
        B2 = chip.rram_bandwidth_per_cycle

        if is_MHA:
            # 多头时，每个头会单独放在一个PE上进行计算
            n_head_per_pe = math.ceil(self.input1_shape[1] / NUM_PES)
            B1 = B1 / NUM_PES
            M = self.input1_shape[2]
            N = self.N
            K = self.K
            res = self.simulate_multicore_mapping(M, N, K, B1, B2, 1, is_dynamic=True)
            res["Cycles"] = n_head_per_pe * res["Cycles"]
            res["Components"] = [c * n_head_per_pe for c in res["Components"]]
        else:
            M = self.M
            N = self.N
            K = self.K  
            res = self.simulate_multicore_mapping(M, N, K, B1, B2, NUM_PES, is_dynamic=False)

        print(f"matmul latency components: SRAM: {res['Components'][0]}, RRAM: {res['Components'][1]}, MME: {res['Components'][2]}, Vector: {res['Components'][3]}")
        return res["Cycles"], res["Components"], res["Loop Order"], res.get("Tm"), res.get("Tn"), res.get("Tk")
        
class softmax:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output

    def mapping_and_simulate(self, chip: chip) :
        # 对于softmax，每个头会单独放在一个PE上进行计算
        n_head = math.ceil(self.input_shape[1]/chip.n_pe)
        M = self.input_shape[2] * n_head
        N = self.N
        latency = 0
        element_size = (self.M * self.N) * self.data_type.word_size
        ldst_latency = math.ceil(element_size / (chip.sram_bandwidth_per_cycle / 4))
        latency += ldst_latency

        cycle_per_exp = chip.pe.vector_unit.cycle_per_exp
        cycle_per_reciprocal = chip.pe.vector_unit.cycle_per_reciprocal
        vector_unit_width = chip.pe.vector_unit.vector_width
        #PE内寻找最大值, 
        find_max_delay = vector_unit_width + M * math.ceil(N / vector_unit_width)
        #减最大值、计算exp函数并向后累加
        exp_accumulation_delay = vector_unit_width  + (1 + cycle_per_exp + 1) * M * math.ceil(N / vector_unit_width)
        #计算指数和的倒数
        reduce_sum_delay = cycle_per_reciprocal
        #逐元素乘
        elementwise_mul_delay = vector_unit_width + M * math.ceil(N / vector_unit_width)

        vector_latency = find_max_delay + exp_accumulation_delay + reduce_sum_delay + elementwise_mul_delay
        latency += vector_latency
        print(f"softmax latency components: SRAM: {ldst_latency}, RRAM: 0, MME: 0, Vector: {vector_latency}")
        return latency, [ldst_latency, 0, 0, vector_latency], None, None, None, None
    
class layernorm:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def mapping_and_simulate(self, chip: chip) :
        # M维度切分到每个PE上
        M = math.ceil(self.M / chip.n_pe)
        N = self.N
        latency = 0
        element_size = (M * N) * self.data_type.word_size
        ldst_latency = math.ceil(element_size / (chip.sram_bandwidth_per_cycle / 4))
        latency += ldst_latency

        cycle_per_reciprocal_sqrt = chip.pe.vector_unit.cycle_per_reciprocal_sqrt
        vector_unit_width = chip.pe.vector_unit.vector_width
        #求和, reduce求和
        cal_mean_value_latency = vector_unit_width + M * math.ceil(N / vector_unit_width)
        #计算方差，减去均值、平方、乘1/d
        cal_variance_latency = vector_unit_width + 3 * M * math.ceil(N / vector_unit_width)
        #归一化，缩放和平移
        cal_normalization_latency = vector_unit_width + (3 + cycle_per_reciprocal_sqrt) * M * math.ceil(N / vector_unit_width)

        vector_latency = cal_mean_value_latency + cal_variance_latency + cal_normalization_latency
        latency += vector_latency
        print(f"layernorm latency components: SRAM: {ldst_latency}, RRAM: 0, MME: 0, Vector: {vector_latency}")
        return latency, [ldst_latency, 0, 0, vector_latency], None, None, None, None

class gelu:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output

    def mapping_and_simulate(self, chip: chip) :
        # N维度切分到每个PE上
        M = math.ceil(self.M / chip.n_pe)  
        N = self.N
        latency = 0
        element_size = (M * N) * self.data_type.word_size
        ldst_latency = math.ceil(element_size / (chip.sram_bandwidth_per_cycle / 4))
        latency += ldst_latency

        # 乘上-1.702，计算exp，加1，取倒数，乘输入
        cycle_per_exp = chip.pe.vector_unit.cycle_per_exp
        cycle_per_reciprocal_sqrt = chip.pe.vector_unit.cycle_per_reciprocal_sqrt
        vector_unit_width = chip.pe.vector_unit.vector_width
        cal_gelu_latency = vector_unit_width + (3 + cycle_per_exp + cycle_per_reciprocal_sqrt) * M * math.ceil(N / vector_unit_width)
        latency += cal_gelu_latency

        print(f"gelu latency components: SRAM: {ldst_latency}, RRAM: 0, MME: 0, Vector: {cal_gelu_latency}")
        return latency, [ldst_latency, 0, 0, cal_gelu_latency], None, None, None, None
    
class res_add:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input1: Tensor ,input2: Tensor=Tensor([1,1],data_type_dict["fp16"])) -> Tensor:
        self.input_shape = input1.shape
        self.output_shape = input1.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def mapping_and_simulate(self, chip: chip) :
        # 切M维度
        M = math.ceil(self.M / chip.n_pe)
        N = self.N
        latency = 0
        element_size = 2 * (M * N) * self.data_type.word_size # 两个输入张量的元素大小之和
        ldst_latency = math.ceil(element_size / (chip.sram_bandwidth_per_cycle / 4))
        latency += ldst_latency

        vector_unit_width = chip.pe.vector_unit.vector_width
        cal_res_add_latency = vector_unit_width + M * math.ceil(N / vector_unit_width)
        latency += cal_res_add_latency

        print(f"res_add latency components: SRAM: {ldst_latency}, RRAM: 0, MME: 0, Vector: {cal_res_add_latency}")
        return latency, [ldst_latency, 0, 0, cal_res_add_latency], None, None, None, None
    
class all_reduce:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def mapping_and_simulate(self, chip: chip) :
        # 4个(M,N)维度的矩阵，切M维度
        M = math.ceil(self.M / chip.n_pe)
        N = self.N
        latency = 0
        element_size = 4 * (M * N) * self.data_type.word_size
        ldst_latency = math.ceil(element_size / (chip.sram_bandwidth_per_cycle / 4))
        latency += ldst_latency

        vector_unit_width = chip.pe.vector_unit.vector_width
        cal_all_reduce_latency = vector_unit_width + 3 * M * math.ceil(N / vector_unit_width)
        latency += cal_all_reduce_latency

        print(f"all_reduce latency components: SRAM: {ldst_latency}, RRAM: 0, MME: 0, Vector: {cal_all_reduce_latency}")
        return latency, [ldst_latency, 0, 0, cal_all_reduce_latency], None, None, None, None