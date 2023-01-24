import csv
import numpy as np
from matplotlib import pyplot as plt
"""
e_0.append(e[0][0])

e_3.append(tau[0])
e_6.append(taud[0])
e_9.append(y[0])
e_12.append(taus0[0])
e_15.append(K@s[0])
e_18.append(taus1[0])
e_21.append(wn[0])

e_24.append(e[0][1])
e_27.append(s[0])
e_30.append(D[0][0])

e_33.append(beta.T @ omega)
"""
n_seed = 0
alpha_lambda = 0.0
alpha_wn0 = 100
alpha_wn1 = 10

alpha_0s0 = 5.0
alpha_0s1 = 5.0
alpha_0s2 = 5.0

alpha_1s0 = 5.0
alpha_1s1 = 5.0
alpha_1s2 = 5.0

T = 1000
step = 0.0001
end = 100

end_plt = 100
start_plt = 0

dir_base = "./data/bzd/"

t_data = np.loadtxt(dir_base + f"step{step}_t{end}.csv")

e_all_p = np.loadtxt(dir_base + f"m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_0s0}_{alpha_0s1}_{alpha_0s2}_T{T}_step{step}_t{end}_norm.csv",delimiter = ",")
# e_all_c = np.loadtxt(dir_base + f"m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_1s0}_{alpha_1s1}_{alpha_1s2}_T{T}_step{step}_t{end}_norm.csv",delimiter = ",")

fig, axes = plt.subplots(nrows=10, ncols=10, sharex=False)

for i in range(10):
    
    for j in range(10):
        # axes[i,j].plot(t_data, e_all_c[3*i+j])
        axes[i,j].plot(t_data, e_all_p[10*i+j])
        
        # axes[i,j].plot(t_data, e_all_c[3*i+j], color="tab:green", label = "Conventional")
        # axes[i,j].plot(t_data, e_all_p[3*i+j], color="tab:red", label = "Proposed")
        # axes[i,j].legend()
        
        axes[i,j].grid()

# plt.savefig(f"abrfwnn/data_test/s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_all.png")

# plt.plot(t_data, e_all_p)

plt.show()
