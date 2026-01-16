import pandas as pd
from software_model.ViT import ViT
from hardware_model.chip import chip_dict
from utils import data_type_dict, Tensor
import os

def run_inference():
    # 1. Initialize
    datatype = data_type_dict["int8"]
    vit = ViT(datatype)
    chip = chip_dict["dcim_chip"]
    
    # 模拟输入以初始化算子形状
    input_tensor = Tensor([1, 196, 768], datatype)
    _ = vit(input_tensor)

    # 2. Run simulation
    operator_latency, total_latency = vit.mapping_and_simulate(chip)

    # 3. Prepare Sheet 1: Operator Latency
    df_ops = pd.DataFrame(operator_latency)
    # Add a row for total
    total_row = {
        "Name": "Total",
        "Total Latency": total_latency,
        "SRAM Latency": df_ops["SRAM Latency"].sum(),
        "RRAM Latency": df_ops["RRAM Latency"].sum(),
        "MME Latency": df_ops["MME Latency"].sum(),
        "Vector Latency": df_ops["Vector Latency"].sum(),
        "Best Loop Order": "",
        "Best tile sizes": ""
    }
    df_ops = pd.concat([df_ops, pd.DataFrame([total_row])], ignore_index=True)

    # 4. Prepare Sheet 2: Hardware Configuration
    hw_config = {
        "Parameter": [
            "Number of PEs",
            "SRAM Bandwidth (GB/s)",
            "Frequency (GHz)",
            "PE RRAM Bandwidth (GB/s)",
            "MME Array Height",
            "MME Array Width",
            "Vector Unit Width",
            "Vector Unit Exp Cycles",
            "Vector Unit Reciprocal Cycles",
            "Vector Unit Reciprocal Sqrt Cycles"
        ],
        "Value": [
            chip.n_pe,
            chip.sram_bandwidth,
            chip.frequency,
            chip.pe.rram.bandwidth,
            chip.pe.mme.array_height,
            chip.pe.mme.array_width,
            chip.pe.vector_unit.vector_width,
            chip.pe.vector_unit.cycle_per_exp,
            chip.pe.vector_unit.cycle_per_reciprocal,
            chip.pe.vector_unit.cycle_per_reciprocal_sqrt
        ]
    }
    df_hw = pd.DataFrame(hw_config)

    # 5. Save to Excel
    output_file = "ViT_inference_latency.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        df_ops.to_excel(writer, sheet_name="Operator Latency", index=False)
        df_hw.to_excel(writer, sheet_name="Hardware Configuration", index=False)

    print(f"Results saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    run_inference()
