import time
import numpy as np
import scipy.sparse as sp
import csv
import torch
num_cores = 1
torch.set_num_threads(num_cores)

def mat_chain(A, B, C):
    return A @ B @ C

def mat_chain_back(A, B, C):
    return C @ B @ A

def mat_elementwise(A, B, C):
    return A * B * C

N = 2000
A_density = .1
B_density = .1
densities = [.1, .01, .001, .0001, .00001]
# 1 Core: 1.9733909130096436
# 4 Cores: 0.5908640384674072
# 8 Cores: 0.46388835906982423
# 16 Cores: 0.35561423301696776

A = torch.from_numpy((sp.random(N,N, A_density)).astype(float).todense()).to_sparse_csc()
B = torch.from_numpy((sp.random(N,N, B_density)).astype(float).todense()).to_sparse_csc()
A_dense = torch.from_numpy(np.random.rand(N,N).astype(float))
B_dense = torch.from_numpy(np.random.rand(N,N).astype(float))
C_dense = torch.from_numpy((np.random.rand(N, int(N/400))).astype(float))
forward_times = []
backward_times =[]
sum_times = []
elementwise_times = []
dense_times = []
for d in densities:
    C = torch.from_numpy(sp.random(N,N, d).astype(float).todense()).to_sparse_csc()

    scripted_mat_chain = torch.jit.script(mat_chain, example_inputs=[(A, B, C)])
    scripted_mat_chain_back = torch.jit.script(mat_chain_back, example_inputs=[(A, B, C)])
    scripted_mat_elementwise = torch.jit.script(mat_chain_back, example_inputs=[(A_dense, B_dense, C)])
    scripted_mat_chain_dense = torch.jit.script(mat_chain, example_inputs=[(A_dense, B_dense, C_dense)])

    n_reps = 7
    avg_time_forward = 0
    avg_time_backward = 0
    avg_time_sum = 0
    avg_time_elementwise = 0
    avg_time_dense = 0
    for i in range(n_reps):
        if i == 2:
            avg_time_forward = 0
            avg_time_backward = 0
            avg_time_sum = 0
            avg_time_elementwise = 0
            avg_time_dense = 0

        start = time.time()
        E1 = scripted_mat_chain(A, B, C)
        end = time.time()
        avg_time_forward = avg_time_forward + end-start
        
        start = time.time()
        E2 = scripted_mat_chain_back(A, B, C)
        end = time.time()
        avg_time_backward = avg_time_backward + end-start

        start = time.time()
        E2 = torch.sparse.sum(scripted_mat_chain(A, B, C).to_sparse_coo())
        end = time.time()
        avg_time_sum = avg_time_sum + end-start

        start = time.time()
        E3 = scripted_mat_elementwise(A_dense, B_dense, C)
        end = time.time()
        avg_time_elementwise = avg_time_elementwise + end-start

        start = time.time()
        E4 = scripted_mat_chain_dense(A_dense, B_dense, C_dense)
        end = time.time()
        avg_time_dense = avg_time_dense + end-start

    avg_time_forward = avg_time_forward / (n_reps - 2) 
    avg_time_backward = avg_time_backward / (n_reps - 2) 
    avg_time_sum = avg_time_sum / (n_reps - 2) 
    avg_time_elementwise = avg_time_elementwise / (n_reps - 2) 
    avg_time_dense = avg_time_dense / (n_reps - 2) 
    forward_times.append(avg_time_forward)
    backward_times.append(avg_time_backward)
    sum_times.append(avg_time_sum)
    elementwise_times.append(avg_time_elementwise)
    dense_times.append(avg_time_dense)
    print("Density: ", d)
    print("Forward Time: ", avg_time_forward)
    print("Backward Time: ", avg_time_backward)
    print("Sum Time: ", avg_time_sum)
    print("Elementwise Time: ", avg_time_elementwise)
    print("Dense Time: ", avg_time_dense)

method = "PyTorch (1 Core)"
if num_cores > 1:
    method = "PyTorch (24 Core)"

data = [("Method", "Algorithm", "Sparsity", "Runtime")]
for i in range(len(densities)):
    data.append((method, "ABC", densities[i], forward_times[i]))
    data.append((method, "CBA", densities[i], backward_times[i]))
    data.append((method, "SUM(ABC)", densities[i], sum_times[i]))
    data.append((method, "A*B*C", densities[i], elementwise_times[i]))
    data.append((method, "ABC Dense", densities[i], dense_times[i]))

filename = "Experiments/Results/mat_exps_pytorch_serial.csv"
if num_cores > 1:
    filename = "Experiments/Results/mat_exps_pytorch_parallel.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the data rows
    csvwriter.writerows(data)

print(forward_times)
print(backward_times)
print(sum_times)
print(elementwise_times)