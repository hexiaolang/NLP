
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import numpy as np

# 读入小说全文
with open('snowNLPtext.txt', 'r') as f:
    text = f.readlines()
    para = [i.rstrip('\n') for i in text]

# 对para中每一段进行情感倾向分析
senti = [SnowNLP(i).sentiments for i in para]

# 横坐标为段落编号，纵坐标为分值，绘制散点图
x = np.array(range(len(senti)))
y = np.array(senti)
plt.scatter(x, y)
plt.show()