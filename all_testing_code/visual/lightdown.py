import matplotlib.pyplot as plt
import numpy as np

# 密底数范围
base_powers = np.linspace(0.1, 1, 10)

# 幂指数范围
exponents = np.linspace(0.4, 0.6, 100)

# 初始化放大倍数结果列表
magnification_factors = []

# 计算放大倍数
for base_power in base_powers:
    magnification_factors_for_base = []
    for exponent in exponents:
        magnification_factor = (base_power ** exponent)/base_power
        magnification_factors_for_base.append(magnification_factor)
    magnification_factors.append(magnification_factors_for_base)

# 绘图
plt.figure(figsize=(10, 6))
for i, base_power in enumerate(base_powers):
    plt.plot(exponents, magnification_factors[i], label=f'Base Power: {base_power:.2f}')

plt.title('Magnification Factors for Different Base Powers')
plt.xlabel('Exponent')
plt.ylabel('Magnification Factor')
plt.legend()
plt.grid(True)

# 保存图像文件
plt.savefig('magnification_factors_plot.png')
plt.close()
