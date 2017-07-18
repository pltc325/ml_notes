import numpy as  np
import matplotlib.pyplot as plt


def f(xx1, xx2):
    return 10 * (xx1 ** 2) + (xx2 ** 2) / 2


def g_unit(xx1, xx2):
    g_x1, g_x2 = 20 * xx1, xx2
    g_norm = np.sqrt((20 * xx1) ** 2 + (xx2) ** 2)
    return g_x1 / g_norm, g_x2 / g_norm, g_norm


def line_search(x1,x2,t,alpha,beta):
    f_current = f(x1,x2)
    g_unit_x1, g_unit_x2, g_norm = g_unit(x1,x2)
    f_new = f(x1 - t*g_unit_x1, x2 - t*g_unit_x2)
    while f_new > f_current + alpha * t * np.sqrt(g_norm):
        t *= beta
        f_new = f(x1 - t*g_unit_x1, x2 - t*g_unit_x2)
    return t


def plot(min_x1,max_x1,min_x2,max_x2,x1_start,x2_start,t,step_num,plot_no,is_line_search,alpha=0.5,beta=0.8):
    plt.subplot(3,1,plot_no)
    x1 = np.arange(min_x1, max_x1, 1)
    x2 = np.arange(min_x2, max_x2, 1)
    xx1, xx2 = np.meshgrid(x1, x2)
    f_values = f(xx1, xx2)
    h = plt.contour(x1, x2, f_values)
    plt.clabel(h, inline=1, fontsize=10)

    x1 = x1_start
    x2 = x2_start
    x1s = np.zeros(step_num)
    x2s = np.zeros(step_num)
    plt.title("t=%.2f and step_num=%d" % (t, step_num))
    for i in range(step_num):
        t_c = t
        x1s[i] = x1
        x2s[i] = x2
        if is_line_search:
            t_c = line_search(x1,x2,t,alpha,beta)
        delta_x1 = -t_c * g_unit(x1, x2)[0]
        delta_x2 = -t_c * g_unit(x1, x2)[1]
        x1 += delta_x1
        x2 += delta_x2
    plt.plot(x1s, x2s, color="red")
    plt.scatter(0, 0, color="yellow")

if __name__ == "__main__":
    plt.figure(figsize=(5,16))
    plot(min_x1=-75,max_x1=75,min_x2=-75,max_x2=75,x1_start=-1,x2_start=20,t=100,step_num=1000,plot_no=1,is_line_search=False)
    plot(min_x1=-20,max_x1=20,min_x2=-20,max_x2=20,x1_start=-1,x2_start=20,t=0.01,step_num=1000,plot_no=2,is_line_search=False)
    plot(min_x1=-20,max_x1=20,min_x2=-20,max_x2=20,x1_start=-1,x2_start=20,t=1,step_num=22,plot_no=3,is_line_search=True,alpha=0.5,beta=0.8)
    plt.savefig('line_search_comparison.png')
    plt.show()
