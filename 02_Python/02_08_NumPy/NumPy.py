import numpy as np
noten = [10, 55, 84, 785, 2154]

noten_np = np.array(noten, dtype=np.int16)
print(noten_np)

listen_min = np.min(noten_np)
listen_max = np.max(noten_np)
print(listen_max)
print(listen_min)

listen_arg_max = np.argmax(noten_np)
listen_arg_min = np.argmin(noten_np)
print(listen_arg_max)
print(listen_arg_min)

listen_mean=np.mean(noten_np)
listen_median=np.median(noten_np)
print(listen_mean)
print(listen_median)