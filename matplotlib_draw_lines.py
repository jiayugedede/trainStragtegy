import matplotlib.pyplot as plt
import numpy as np

# 没置随机数种子以便结果可复现np,random,seed(0)
# 置岛形尺寸
plt.figure(figsize=(19，6))
# 没置债华标和别华标的范围
x = np.arange(8.15, 8.35， 9.825
y = np.logspace(-4，-1， len(x))

# 定义额色、线型和标记
colors = ['b',gm'1linestyles =markers = ['o'D'
# 生成几条随机折线for i in range(5):plt.plot(x, y * np.random.rand(len(x)), label=f'Line {i+1)', color=colors[i % len(colors)], linestyle = linestyles[i % len(linestyles)], marker=markers[i % len(markers)], linewidth=2)

# 战置纵些标为对数华标
plt.yscale("log')
#旅加网楼
plt.grid(True, which="both", s="-", color='8.7')
# 没置标题和华标船标签
plt,title('Random Lines in Logarithmic Scale')
plt,xlabel("X-axis')
plt.ylabel("Y-axis (log scale)')
# 没置岛例
plt.legend()
# 显示图形
plt.show()