import math
import itertools

def simulate_multicore_mapping(M, N, K, B1, B2):
    """
    M, N, K: 全局矩阵大小
    B1: Shared SRAM 带宽 (elements/cycle/bank_group) - 假设广播只占1个带宽
    B2: Distributed RRAM 带宽 (elements/cycle/pe) - 每个PE独享
    """
    NUM_PES = 4
    
    # 硬件参数 (单个PE的Tile)
    Tm, Tn, Tk = 32, 32, 72
    size_tile_a = Tm * Tk
    size_tile_b = Tk * Tn
    size_tile_c = Tm * Tn * 4 #input是1byte，output是4byte
    compute_cycles = Tm
    
    results = []

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
                        
                        # 1. 收集所有PE在当前时刻对 SRAM (A, C/D) 和 RRAM (B) 的请求
                        load_a_addr = [] # load A 的逻辑块坐标
                        load_c_addr = [] # load C 的逻辑块坐标
                        store_d_addr = [] # store D 的逻辑块坐标
                        load_b_occurred = False # 只要有一个PE读B，就是RRAM开销(并行)
                        
                        current_step_sram_load_vol = 0
                        current_step_sram_store_vol = 0
                        current_step_rram_vol = 0
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
                            
                            # --- 分析 B (RRAM) ---
                            curr_b = (global_k_idx, global_n_idx)
                            if curr_b != prev_b_indices[pe_id]:
                                # RRAM是独立的，每个PE自己读自己的，带宽不竞争
                                # 瓶颈取决于最慢的那个PE（这里是对称的，所以就是单次加载时间）
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
                        
                        # SRAM 读写耗时
                        time_sram_load = math.ceil((cost_load_a + cost_load_c) / B1)
                        time_sram_store = math.ceil((cost_store_d) / B1)
                        
                        # 3. RRAM B 的开销
                        # 因为是分布式的，且并行，只要发生了加载，耗时就是一个Tile的时间
                        time_rram = math.ceil((size_tile_b / B2)) if load_b_occurred else 0
                        
                        # 4. 计算核心耗时 (流水线瓶颈)
                        step_latency = max(time_sram_load, time_sram_store, time_rram, compute_cycles)
                        total_cycles += step_latency
            
            # 加上最后的Flush时间 (C写回)
            # 假设写回不冲突情况下的理想带宽，或者算上冲突
            # 这里简化处理：所有PE都要写回最后一块，互不相同
            flush_cost = (NUM_PES * size_tile_c) / B1
            total_cycles += flush_cost

            results.append({
                "Strategy": strat['name'],
                "Loop Order": "->".join(order) + " (Inner)",
                "Cycles": int(total_cycles)
            })

    results.sort(key=lambda x: x['Cycles'])
    return results
if __name__ == "__main__":
    # --- 运行测试 ---
    # 场景：4个PE，SRAM带宽B1=64 (共享), RRAM带宽B2=16 (独占)
    # 大矩阵
    M, N, K = 197, 384, 384
    B1_shared = 512
    B2_private = 32

    perf_data = simulate_multicore_mapping(M, N, K, B1_shared, B2_private)

    print(f"{'Strategy':<30} | {'Loop Order':<20} | {'Cycles':<10}")
    print("-" * 70)
    for p in perf_data[:12]: # 打印前12名
        print(f"{p['Strategy']:<30} | {p['Loop Order']:<20} | {p['Cycles']:<10}")
