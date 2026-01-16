import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from software_model.operator import matmul, softmax, layernorm, gelu, res_add, all_reduce
from software_model.misc import quantization ,reshape ,split ,transpose ,Concat
from hardware_model.chip import chip, chip_dict

class ViT():
    def __init__(self, 
                datatype: DataType, 
                patch_size = 768 ,
                hidden_dim = 384 ,
                head_dim = 64 ,
                n_attn_heads = 6 ,
                intermediate_size= 1536,
                n_layers = 12 ,
                TP=1
        ):
        self.datatype = datatype
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_attn_heads = n_attn_heads
        self.intermediate_size = intermediate_size
        self.n_layers = n_layers
        self.TP = TP

        self.embedding = Tensor([self.patch_size, self.hidden_dim], self.datatype)
        #attention weights
        self.WQ = Tensor([self.hidden_dim, self.head_dim*self.n_attn_heads], self.datatype)
        self.WK = Tensor([self.hidden_dim, self.head_dim*self.n_attn_heads], self.datatype)
        self.WV = Tensor([self.hidden_dim, self.head_dim*self.n_attn_heads], self.datatype)
        self.WO = Tensor([self.head_dim*self.n_attn_heads, self.hidden_dim], self.datatype)

        #ffn weights
        self.W_linear_up = Tensor([self.hidden_dim, self.intermediate_size], self.datatype)
        self.W_linear_down = Tensor([self.intermediate_size, self.hidden_dim], self.datatype)
        
        self.patch_embedding = matmul(self.datatype)
        #attention
        self.attn_layernorm = layernorm(data_type=data_type_dict["fp32"])
        self.Q_proj = matmul(self.datatype)
        self.K_proj = matmul(self.datatype)
        self.V_proj = matmul(self.datatype)
        self.Q_reshape = reshape(self.datatype)
        self.K_reshape = reshape(self.datatype)
        self.V_reshape = reshape(self.datatype)
        self.K_transpose = transpose(self.datatype)
        self.QKT = matmul(self.datatype)
        self.softmax = softmax(data_type=data_type_dict["fp32"])
        self.SV = matmul(self.datatype)
        self.SV_reshape = reshape(self.datatype)
        self.O_proj = matmul(self.datatype)
        self.attn_all_reduce = all_reduce(data_type=data_type_dict["fp32"])
        self.attn_resadd = res_add(data_type=data_type_dict["fp32"])
        
        #ffn
        self.ffn_layernorm = layernorm(data_type=data_type_dict["fp32"])
        self.ffn_linear_up = matmul(self.datatype)
        self.ffn_gelu = gelu(data_type=data_type_dict["fp32"])
        self.ffn_linear_down = matmul(self.datatype)
        self.ffn_resadd = res_add(data_type=data_type_dict["fp32"])
    
    def __call__(self, input: Tensor) -> Tensor:
        b, s, d = input.shape
        assert d == self.patch_size
        input = self.patch_embedding(input, self.embedding)
        cls_token = Tensor([1,1,self.hidden_dim], self.datatype)
        concat = Concat(data_type=self.datatype)
        input = concat([cls_token, input], dim=1)
        #attention
        input = self.attn_layernorm(input)
        q = self.Q_proj(input, self.WQ)
        q = self.Q_reshape(q, [b, self.n_attn_heads, s+1, self.head_dim])
        k = self.K_proj(input, self.WK)
        k = self.K_reshape(k, [b, self.n_attn_heads, s+1, self.head_dim])
        k = self.K_transpose(k, [0 ,1 ,3, 2])
        v = self.V_proj(input, self.WV)
        v = self.V_reshape(v, [b, self.n_attn_heads, s+1, self.head_dim])
        qkT = self.QKT(q, k)
        Score = self.softmax(qkT)
        SV = self.SV(Score, v)
        SV = self.SV_reshape(SV, [b, s+1, self.n_attn_heads*self.head_dim])
        O = self.O_proj(SV, self.WO)
        O = self.attn_all_reduce(O)
        attn_output = self.attn_resadd(input, O)
        #ffn
        ffn_input = self.ffn_layernorm(attn_output)
        ffn_output = self.ffn_linear_up(ffn_input, self.W_linear_up)
        ffn_output = self.ffn_gelu(ffn_output)
        ffn_output = self.ffn_linear_down(ffn_output, self.W_linear_down)
        output = self.ffn_resadd(attn_output, ffn_output)
        return output
    
    def mapping_and_simulate(self, chip: chip) :
        operator_latency = []
        total_latency = 0

        def add_op(name, res):
            nonlocal total_latency
            latency, components, loop_order, Tm, Tn, Tk = res
            operator_latency.append({
                "Name": name,
                "Total Latency": latency,
                "SRAM Latency": components[0],
                "RRAM Latency": components[1],
                "MME Latency": components[2],
                "Vector Latency": components[3],
                "Best Loop Order": loop_order,
                "Best tile sizes": (Tm, Tn, Tk) if Tm is not None and Tn is not None and Tk is not None else None
            })
            total_latency += latency

        add_op("patch_embedding", self.patch_embedding.mapping_and_simulate(chip, is_MHA=False))
        for i in range(self.n_layers):
            add_op(f"layer_{i}_attn_layernorm", self.attn_layernorm.mapping_and_simulate(chip))
            add_op(f"layer_{i}_Q_proj", self.Q_proj.mapping_and_simulate(chip, is_MHA=False))
            add_op(f"layer_{i}_K_proj", self.K_proj.mapping_and_simulate(chip, is_MHA=False))
            add_op(f"layer_{i}_V_proj", self.V_proj.mapping_and_simulate(chip, is_MHA=False))
            add_op(f"layer_{i}_QKT", self.QKT.mapping_and_simulate(chip, is_MHA=True))
            add_op(f"layer_{i}_softmax", self.softmax.mapping_and_simulate(chip))
            add_op(f"layer_{i}_SV", self.SV.mapping_and_simulate(chip, is_MHA=True))
            add_op(f"layer_{i}_O_proj", self.O_proj.mapping_and_simulate(chip, is_MHA=False))
            add_op(f"layer_{i}_attn_all_reduce", self.attn_all_reduce.mapping_and_simulate(chip))
            add_op(f"layer_{i}_attn_resadd", self.attn_resadd.mapping_and_simulate(chip))
            add_op(f"layer_{i}_ffn_layernorm", self.ffn_layernorm.mapping_and_simulate(chip))
            add_op(f"layer_{i}_ffn_linear_up", self.ffn_linear_up.mapping_and_simulate(chip, is_MHA=False))
            add_op(f"layer_{i}_ffn_gelu", self.ffn_gelu.mapping_and_simulate(chip))
            add_op(f"layer_{i}_ffn_linear_down", self.ffn_linear_down.mapping_and_simulate(chip, is_MHA=False))
            add_op(f"layer_{i}_ffn_resadd", self.ffn_resadd.mapping_and_simulate(chip))
        
        return operator_latency, total_latency

if __name__ == "__main__":
    ViT_inference = ViT(data_type_dict["int8"])
    chip = chip_dict["dcim_chip"]
    input = Tensor([1, 196, 768], data_type_dict["int8"])
    output = ViT_inference(input)
    operator_latency, total_latency = ViT_inference.mapping_and_simulate(chip)
    print("Operator Latency:", operator_latency)
    print("Total Latency:", total_latency)