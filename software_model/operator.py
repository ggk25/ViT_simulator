import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.chip import chip ,chip_dict
import math

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
    def simulate_pe_latency(self, chip: chip) -> float:
        # Calculate the number of operations
        num_operations = self.M * self.N * self.K
        # Calculate the latency based on the PE's frequency and number of PEs
        latency = num_operations / (chip.n_pe * chip.pe.frequency)
        return latency
    
    def mapping_and_simulate(self, chip: chip) -> float:
        # Map the operation to the chip and simulate the latency
        latency = self.simulate_pe_latency(chip)
        return latency

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
    
    def simulate_pe_latency(self, chip: chip) -> float:
        # Calculate the number of operations
        num_operations = self.M * self.N
        # Calculate the latency based on the PE's frequency and number of PEs
        latency = num_operations / (chip.n_pe * chip.pe.frequency)
        return latency
    
    def mapping_and_simulate(self, chip: chip) -> float:
        # Map the operation to the chip and simulate the latency
        latency = self.simulate_pe_latency(chip)
        return latency
    
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
    def simulate_pe_latency(self, chip: chip) -> float:
        # Calculate the number of operations
        num_operations = self.M * self.N
        # Calculate the latency based on the PE's frequency and number of PEs
        latency = num_operations / (chip.n_pe * chip.pe.frequency)
        return latency
    
    def mapping_and_simulate(self, chip: chip) -> float:
        # Map the operation to the chip and simulate the latency
        latency = self.simulate_pe_latency(chip)
        return latency

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
    def simulate_pe_latency(self, chip: chip) -> float:
        # Calculate the number of operations
        num_operations = self.M * self.N
        # Calculate the latency based on the PE's frequency and number of PEs
        latency = num_operations / (chip.n_pe * chip.pe.frequency)
        return latency
    
    def mapping_and_simulate(self, chip: chip) -> float:
        # Map the operation to the chip and simulate the latency
        latency = self.simulate_pe_latency(chip)
        return latency
    
class element_wise_mul_add:
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
    def simulate_pe_latency(self, chip: chip) -> float:
        # Calculate the number of operations
        num_operations = self.M * self.N
        # Calculate the latency based on the PE's frequency and number of PEs
        latency = num_operations / (chip.n_pe * chip.pe.frequency)
        return latency
    
    def mapping_and_simulate(self, chip: chip) -> float:
        # Map the operation to the chip and simulate the latency
        latency = self.simulate_pe_latency(chip)
        return latency