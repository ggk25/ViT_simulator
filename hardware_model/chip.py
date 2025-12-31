from pe import pe, pe_dict

class chip:
    def __init__(
        self,
        n_pe,
        pe: pe,
        sram_bandwidth,
        frequency,
    ):
        self.n_pe = n_pe
        self.pe = pe
        self.sram_bandwidth = sram_bandwidth  # GB/s
        self.frequency = frequency  # GHz
        self.sram_bandwidth_per_cycle = (sram_bandwidth / frequency)  # B/cycle
        self.rram_bandwidth_per_cycle = (pe.rram.bandwidth / frequency)  # B/cycle
chip_dict = {
    "dcim_chip": chip(
        n_pe=4,
        pe=pe_dict["dcim"],
        sram_bandwidth=128,  # GB/s
        frequency=0.5,  # GHz
    ),
}