import csv
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

# xold = []

# for i in range(5):
#     xold.append(i * np.arange(6).reshape(3,2))

# print(xold)
# print(xold[3])
# print(xold[3][0][1])

# if np.abs(e[0][0]) < np.abs(xold[-10000][0][0]):

a = np.arange(6).reshape(3,2) - 3

print(a)
print(a**(3*np.ones((3,2))-1))
print(np.sign(a))
print(np.sign(a)**(3/2))