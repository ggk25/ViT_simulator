import math
import itertools
from collections import defaultdict

def simulate_multicore_mapping(M, N, K, B1, B2, is_dynamic=False):
    """
    M, N, K: 全局矩阵大小
    B1: Shared SRAM 带宽 (elements/cycle/bank_group) - 假设广播只占1个带宽. 
         (Modify: B1 is total bytes/cycle. 512B/cycle -> 128 Banks x 4B/bank)
    B2: Distributed RRAM 带宽 (elements/cycle/pe) - 每个PE独享
    """
    NUM_PES = 4
    SRAM_BANKS = 128
    BYTES_PER_BANK = 4 # 512 Bytes / 128 Banks = 4 Bytes
    
    # helper for bank conflict calculation
    def get_bank_counts(current_counts, req_tiles, tile_rows, tile_cols, global_width, elem_size):
        # req_tiles: list of (r_idx, c_idx)
        
        # New Layout Strategy: 16x32 Sub-Tiles
        # Map (global_r, global_c) -> Tiled Address
        SUB_TILE_H = 16
        SUB_TILE_W = 32
        
        # 整个大矩阵在行方向上有多少个subtile
        subtile_per_row = (global_width + SUB_TILE_W - 1) // SUB_TILE_W 
        bytes_per_subtile = SUB_TILE_H * SUB_TILE_W * elem_size

        # r_idx和c_idx是逻辑Tile的索引
        for (r_idx, c_idx) in req_tiles:
            # 行列起始地址
            r_start = r_idx * tile_rows
            c_start = c_idx * tile_cols
            
            #遍历tile中的每一行
            for r in range(tile_rows):
                global_r = r_start + r #在整个矩阵中的行id
                

                subtile_r = global_r // SUB_TILE_H #在整个矩阵中的sub-tile行id
                off_r = global_r % SUB_TILE_H #在这个subtile内的行偏移
                
                # 这一个subtile的起始行地址
                row_subtile_base_addr = subtile_r * subtile_per_row * bytes_per_subtile
                
                
                step_elems = 4 // elem_size if elem_size < 4 else 1
                step_bytes = step_elems * elem_size
                
                # 遍历tile中的每一列
                for c in range(0, tile_cols, step_elems):
                    global_c = c_start + c
                    
                    subtile_c = global_c // SUB_TILE_W #在整个矩阵中的sub-tile列id
                    off_c = global_c % SUB_TILE_W  #在这个subtile内的列偏移
                    
                    # 在sub-tile内的偏移地址，以字节为单位
                    offset_in_blk = (off_r * SUB_TILE_W + off_c) * elem_size
                    
                    # subtile的基地址
                    blk_addr = row_subtile_base_addr + subtile_c * bytes_per_subtile
                    
                    # 基地址加偏移
                    final_addr = blk_addr + offset_in_blk
                    
                    # Bank Mapping
                    bank_id = (final_addr // BYTES_PER_BANK) % SRAM_BANKS
                    current_counts[bank_id] += 1

    # 候选硬件参数 (Tm, Tn, Tk)
    candidates = [
        #(32, 32, 72),
        (16, 16, 144),
        #(8, 8, 288),
        #(4, 4, 576),
        #(2, 2, 1152)
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
                total_cycles_ideal = 0
                total_cycles_bank = 0
                
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
                            
                            # 1. 原有的理想带宽计算 (基于总流量)
                            unique_a_requests = set(load_a_addr)
                            unique_b_requests = set(load_b_addr) if is_dynamic else set()
                            unique_load_c_requests = set(load_c_addr)
                            
                            cost_load_a = len(unique_a_requests) * size_tile_a
                            cost_load_c = len(unique_load_c_requests) * size_tile_c
                            cost_store_d = len(store_d_addr) * size_tile_c
                            
                            if is_dynamic:
                                cost_load_b = len(unique_b_requests) * size_tile_b
                                time_rram = 0
                            else:
                                cost_load_b = 0
                                time_rram = math.ceil((size_tile_b / B2)) if load_b_occurred else 0
                            
                            time_sram_load_ideal = math.ceil((cost_load_a + cost_load_b + cost_load_c) / B1)
                            time_sram_store_ideal = math.ceil((cost_store_d) / B1)
                            
                            # 2. 新增：基于 Bank Conflict 的真实带宽计算
                            # 所有的 Load 请求 (A, B-if-dynamic, C) 竞争 SRAM Bank
                            # 所有的 Store 请求 (D) 竞争 SRAM Bank (假设 Load/Store 端口共享或者分时)
                            # 在这里假设 Load 和 Store 不能同时发生，或者争抢同一个 Time Window
                            
                            bank_counts_load = [0] * SRAM_BANKS
                            bank_counts_store = [0] * SRAM_BANKS
                            
                            # A Requests (M x K matrix, 1 byte/elem)
                            # unique_a_requests stores (m_subtile, k_subtile)
                            get_bank_counts(bank_counts_load, list(unique_a_requests), 
                                          Tm, Tk, K, 1)
                                          
                            # C Requests (M x N matrix, 4 bytes/elem)
                            # unique_load_c_requests stores (m_subtile, n_subtile)
                            get_bank_counts(bank_counts_load, list(unique_load_c_requests),
                                          Tm, Tn, N, 4)
                            
                            if is_dynamic:
                                # B Requests (K x N matrix, usually 1 byte) 
                                # ...assuming B matches A prec or C prec? Usually B (weights) is lower prec.
                                # Let's assume B is 1 byte like A.
                                get_bank_counts(bank_counts_load, list(unique_b_requests),
                                              Tk, Tn, N, 1)

                            # D Store Requests (M x N matrix, 4 bytes/elem)
                            get_bank_counts(bank_counts_store, store_d_addr,
                                          Tm, Tn, N, 4)
                            
                            max_load_cycles = max(bank_counts_load) if bank_counts_load else 0
                            max_store_cycles = max(bank_counts_store) if store_d_addr else 0
                            
                            # 最终 Latency
                            step_latency_ideal = max(time_sram_load_ideal, time_sram_store_ideal, time_rram, compute_cycles)
                            
                            # Bank 限制下的 Latency (SRAM Load 和 Store 如果也是共享端口，则取 max(load, store) 还是 sum? 
                            # 即使是双端口，同一个Bank通常也只能一读一写或者冲突。保守起见，如果Load/Store都很重，Bank利用率是关键。
                            # 简单起见，分别计算瓶颈)
                            step_latency_bank = max(max_load_cycles, max_store_cycles, time_rram, compute_cycles)

                            total_cycles_ideal += step_latency_ideal
                            total_cycles_bank += step_latency_bank

                
                # Flush 阶段也受 Bank 限制
                # 假设 Flush 是所有 PE 写回最后一块
                # Flush 理想
                flush_cost_ideal = (NUM_PES * size_tile_c) / B1
                
                # Flush Bank
                # 需要构造 Flush 的请求列表
                # 因为循环结束时 prev_d_indices 保留了最后一组
                # 其实就是最后一次的 store_d 操作，但这已经在循环里处理了吗？
                # 不，循环里的 store_d 是 "Switch Tile" 时触发的。
                # 最后一个 Tile 做完后，没有下一次 Switch，所以需要强制 Flush。
                flush_store_addr = [idx for idx in prev_d_indices if idx is not None]
                flush_bank_counts = [0] * SRAM_BANKS
                get_bank_counts(flush_bank_counts, flush_store_addr, Tm, Tn, N, 4)
                flush_cost_bank = max(flush_bank_counts) if flush_bank_counts else 0

                total_cycles_ideal += flush_cost_ideal
                total_cycles_bank += flush_cost_bank

                results.append({
                    "Strategy": strat['name'],
                    "Loop Order": "->".join(order) + " (Inner)",
                    "Cycles (Ideal)": int(total_cycles_ideal),
                    "Cycles (Bank)": int(total_cycles_bank),
                    "Tm": Tm,
                    "Tn": Tn,
                    "Tk": Tk
                })
        
    results.sort(key=lambda x: x['Cycles (Bank)'])
    return results
if __name__ == "__main__":
    # --- 运行测试 ---
    # 场景：4个PE，SRAM带宽B1=64 (共享), RRAM带宽B2=16 (独占)
    # 大矩阵
    M, N, K = 196, 1536, 384
    B1_shared = 512
    B2_private = 192

    perf_data = simulate_multicore_mapping(M, N, K, B1_shared, B2_private, is_dynamic=False)
    
    print(f"--- Static Mapping (Weights in RRAM) ---")
    print(f"{'Tm':<4} | {'Tn':<4} | {'Tk':<4} | {'Strategy':<30} | {'Loop Order':<20} | {'Ideal':<10} | {'Bank':<10}")
    print("-" * 110)
    for p in perf_data[:15]: # 打印前5名
        print(f"{p['Tm']:<4} | {p['Tn']:<4} | {p['Tk']:<4} | {p['Strategy']:<30} | {p['Loop Order']:<20} | {p['Cycles (Ideal)']:<10} | {p['Cycles (Bank)']:<10}")
    '''
    print(f"\n--- Dynamic Mapping (Weights in SRAM) ---")
    perf_data_dynamic = simulate_multicore_mapping(M, N, K, B1_shared, B2_private, is_dynamic=True)
    print(f"{'Tm':<4} | {'Tn':<4} | {'Tk':<4} | {'Strategy':<30} | {'Loop Order':<20} | {'Cycles':<10}")
    print("-" * 95)
    for p in perf_data_dynamic[:15]: # 打印前5名
        print(f"{p['Tm']:<4} | {p['Tn']:<4} | {p['Tk']:<4} | {p['Strategy']:<30} | {p['Loop Order']:<20} | {p['Cycles']:<10}")
    '''