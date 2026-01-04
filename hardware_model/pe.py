from utils import data_type_dict

class vector_unit:
    def __init__(
        self, 
        word_size,
        cycle_per_exp, # cycles per exp instruction
		cycle_per_reciprocal, # cycles per reciprocal instruction
		cycle_per_reciprocal_sqrt, # cycles per reciprocal_square_root instruction
		cycle_per_vector_loop, # cycles per vector loop
        cycle_per_vector_ldst, # cycles per vector load/store
        vector_width, # vector width
        data_type=data_type_dict["fp32"],
    )-> None:
        self.word_size = word_size  # Byte
        self.cycle_per_vector_ldst = cycle_per_vector_ldst
        self.cycle_per_exp = cycle_per_exp  
        self.cycle_per_reciprocal = cycle_per_reciprocal
        self.cycle_per_reciprocal_sqrt = cycle_per_reciprocal_sqrt
        self.cycle_per_vector_loop = cycle_per_vector_loop
        self.vector_width = vector_width
        self.data_type = data_type

vector_unit_dict = {
    "dcim": vector_unit(word_size=4, cycle_per_exp=15, cycle_per_reciprocal=12, \
                        cycle_per_reciprocal_sqrt=15, cycle_per_vector_loop=1, \
                        cycle_per_vector_ldst=2, vector_width=128),
}

class mme:
    def __init__(
        self,
        array_height,
        array_width,
        mac_per_cycle,
        input_word_size,
        output_word_size,
    ):
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle = mac_per_cycle
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size

mme_dict = {
    "dcim": mme(array_height=32, array_width=32, mac_per_cycle=1, input_word_size=1,\
                output_word_size=4),
}

class rram:
    def __init__(
        self,
        n_macro,  # number of rram macros
        SA_bitwidth,    # SA位宽
        read_latency,   # ns
    ):
        self.n_macro = n_macro
        self.SA_bitwidth = SA_bitwidth
        self.read_latency = read_latency
        self.bandwidth = (SA_bitwidth / 8) * n_macro * 1e3  # GB/s
rram_dict = {
    "dcim": rram(n_macro=32, SA_bitwidth=72, read_latency=18),
}

class pe:
    def __init__(
        self,
        vector_unit: vector_unit,
        mme: mme,
        rram: rram,
    ):
        self.vector_unit = vector_unit
        self.mme = mme
        self.rram = rram
pe_dict = {
    "dcim": pe(vector_unit=vector_unit_dict["dcim"], mme=mme_dict["dcim"], \
               rram=rram_dict["dcim"]),
}

        