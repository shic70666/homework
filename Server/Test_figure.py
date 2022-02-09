#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import figureHB as hb
plt.style.use("figureplot.mplstyle")

df = pd.read_csv("/Users/shic/Codes/personal/Homework/Server/Test_figure_data.csv", header = None)
df.columns = ["Time","Acceleration","Strain","Velocity"] 

#%% 截取前1000条数据，画Time-Acceleration图，所有图形必须有 title;legend;Xlable;Ylable
a_axis = df[:90]["Time"]
b_axis = df[:90]["Acceleration"]
c_axis = df[:90]["Strain"]
d_axis = df[:90]["Velocity"]

plt.figure(figsize=(16,6))
plt.title("Time-Acceleration")
plt.plot(a_axis, b_axis, label="Time-Acceleration")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration $(m^2/s)$")

plt.grid(True)
plt.legend(loc="upper right")

plt.show()

#%%  截取前1000条数据，在一个figure里画 Time-Acceleration 和 Time-Strain的图
plt.figure(figsize=(16,12))
plt.subplot(211)
plt.title("Time-Acceleration")
plt.plot(a_axis, b_axis, label="Time-Acceleration")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration $(m^2/s)$")

plt.grid(True)
plt.legend(loc="upper right")

plt.subplot(212)
plt.title("Time-Strain")
plt.plot(a_axis, c_axis, label="Time-Strain")

plt.xlabel("Time (s)")
plt.ylabel("Strain $(\mu m/m)$")

plt.grid(True)
plt.legend(loc="upper right")

plt.show()

# #%%  
# plt.figure(figsize=(16,12))
# # plt.subplot(211)
# # The above 2 lines equals to:
# figure, axes = plt.subplots(2,1,)

# axes[0].plot(a_axis, b_axis, label="Time-Acceleration")
# axes[1].plot(a_axis, c_axis, label="Time-Strain")
# axes[0].set_ylabel("$Acceleration (m^2/s)$")
# axes[1].set_ylabel("$Strain (\mu m/m)$")

# for axis in axes:
#     axis.legend(loc="upper right")
#     axis.grid()
#     axis.set_xlabel("Time (s)")
#     axis.set_title("common title")

#%% 截取前1000条数据，在一个figure，一个graph里画 Time-Acceleration(红，实线) 和 Time-Velocity(蓝，虚线)的图

plt.figure(figsize=(16,6))

plt.title("Time-Acceleration $\&$ Time-Velocity")
plt.plot(a_axis, b_axis, 'r-', linewidth=2,  label="Time-Acceleration")
plt.plot(a_axis, d_axis, color = 'cyan', linestyle = '--', linewidth=10, alpha=0.5, label="Time-Velocity")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration $(m^2/s)$")

plt.grid(True)
plt.legend(loc="upper right")

plt.show()


# %%
