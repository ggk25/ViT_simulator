程序说明：
    1. operator.py中定义了各个算子的mapping方法，这里的mapping是算子mapping到整个芯片上的延迟；
    2. matmul算子对于非多头矩阵乘，切分权重矩阵的N维度到每个PE上，然后遍历M,N,K的循环顺序，延迟组成包含SRAM访问延迟、RRAM访问延迟、mme计算延迟
    3. operator中的其他算子是element-wise算子，计算在vector中进行，延迟包含SRAM访问延迟和vector计算延迟
TODO：
    1. matmul算子支持硬件阵列尺寸的寻优
    2. vector算力的调整以及延迟计算方式改进，可能也调整为tile粒度的描述
    3. kernel fusion
    4. 卷积支持

2026 1/6
1.动态矩阵乘修改为B也从SRAM取
2.增加tile尺寸变化
todo：换mapping方式