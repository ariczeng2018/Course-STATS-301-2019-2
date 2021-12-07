import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height), ha='center', va='bottom')


plt.plot([1, 2, 3, 10])
plt.xlabel("x- axis")
plt.ylabel("my numbers")
plt.title("my figure")
plt.show()

y = np.random.standard_normal((20, 2)).cumsum(axis=0)
plt.figure(figsize=(7, 4))
plt.plot(y, lw=1.5)  # plots two lines
plt.plot(y, 'ro')  # plots two dotted lines
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 1], lw=1.5, label='2nd')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.show()

# different scales
y[:, 0] = y[:, 0] * 100
plt.figure(figsize=(7, 4))
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 1], lw=1.5, label='2nd')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.show()

fig, ax1 = plt.subplots()
plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=8)
plt.axis('tight')
# 'tight'  Set limits just large
# enough to show all data.
plt.xlabel('index')
plt.ylabel('value 1st')
plt.title('A Simple Plot')
plt.show()

ax2 = ax1.twinx()
plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
plt.plot(y[:, 1], 'ro')
plt.legend(loc=0)
plt.ylabel('value 2nd')
plt.show()

plt.figure(figsize=(7, 5))  # 两个单独子图
plt.subplot(2, 1, 1)
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.subplot(212)
plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
plt.plot(y[:, 1], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.show()

plt.figure(figsize=(9, 4))  # 组合子图
plt.subplot(121)
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('1st Data Set')
plt.subplot(122)
plt.bar(np.arange(len(y)), y[:, 1], width=0.5,
        color='g', label='2nd')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.title('2nd Data Set')
plt.show()

y = np.random.standard_normal((1000, 2))  # scatter plot
plt.figure(figsize=(7, 5))
plt.plot(y[:, 0], y[:, 1], 'ro')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y[:, 0], y[:, 1], marker='o')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.show()

# allows the addition of a third dimension
# Scatter plot with third dimension
c = np.random.randint(0, 10, len(y))
plt.figure(figsize=(7, 5))
plt.scatter(y[:, 0], y[:, 1], c=c, marker='o')
plt.colorbar()
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.show()

# Stacked histogram for two data sets
plt.figure(figsize=(7, 4))
plt.hist(y, label=['1st', '2nd'], bins=25)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Histogram')
plt.show()

# box plot
_, ax = plt.subplots(figsize=(7, 4))
plt.boxplot(y)
plt.grid(True)
plt.setp(ax, xticklabels=['1st', '2nd'])
plt.xlabel('data set')
plt.ylabel('value')
plt.title('Boxplot')
plt.show()

# real examples
pv = 1000
r = 0.08
n = 10
t = np.linspace(0, n, n)
y1 = np.ones(len(t)) * pv  # this is a horizontal line
y2 = pv * (1 + r * t)
y3 = pv * (1 + r) ** t
plt.title('Simple vs. compounded interest rates')
plt.xlabel('Number of years')
plt.ylabel('Values')
plt.xlim(0, 11)
plt.ylim(800, 2200)
plt.plot(t, y1, 'b-'), plt.plot(t, y2, 'g--'), plt.plot(t, y3, 'r-')
plt.show()

ind = np.arange(3)
plt.title("DuPont Identity")
plt.xlabel("Different companies")
plt.ylabel("Three ratios")
ROE = [0.88, 0.22, 0.22]
a = [0.16, 0.04, 0.036]
b = [0.88, 1.12, 2.31]
c = [6.32, 4.45, 2.66]
width = 0.45
plt.figtext(0.2, 0.85, "ROE=0.88")
plt.figtext(0.5, 0.7, "ROE=0.22")
plt.figtext(0.8, 0.6, "ROE=0.22")
plt.figtext(0.2, 0.75, "Profit Margin=0.16")
plt.figtext(0.5, 0.5, "0.041")
plt.figtext(0.8, 0.4, "0.036")
p1 = plt.bar(ind, a, width, color='b', label='profitMargin')
p2 = plt.bar(ind, b, width, color='r', bottom=a, label='assetTurnover')
p3 = plt.bar(ind, c, width, color='y', bottom=b, label='equitMultiplier')
plt.xticks([0., 1., 2.], ['IBM', 'DELL', 'WMT'])
plt.legend(loc='upper right')
plt.show()

# 有效使用颜色
A_EPS = (5.02, 4.54, 4.18, 3.73)
B_EPS = (1.35, 1.88, 1.35, 0.73)
ind = np.arange(len(A_EPS))  # the x locations for the groups
width = 0.40  # the width of the bars
_, ax = plt.subplots()
A_Std = B_Std = (2, 2, 2, 2)
rects1 = ax.bar(ind, A_EPS, width, color='r', yerr=A_Std)
rects2 = ax.bar(ind + width, B_EPS, width, color='y', yerr=B_Std)
ax.set_ylabel('EPS')
ax.set_xlabel('Year')
ax.set_title('Diluted EPS Excluding Extraordinary Items ')
ax.set_xticks(ind + width)
ax.set_xticklabels(('2012', '2011', '2010', '2009'))
ax.legend((rects1[0], rects2[0]), ('W-Mart', 'DELL'))
autolabel(rects1)
autolabel(rects2)
plt.show()
