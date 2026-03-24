import os
import secrets

n_seeds = 200
seeds = []
for i in range(n_seeds):
    seeds.append(secrets.randbits(128))

with open('200_seeds.txt', 'w') as f:
    seed = secrets.randbits(128)
    f.writelines([str(i)+'\n' for i in seeds])