
import argparse 
parser = argparse.ArgumentParser(description="Matrix chain benchmarks (PyTorch)")
parser.add_argument("--parallel", "-p", type=bool, default=True, help="Whether to run in parallel")
parallel = parser.parse_args().parallel

import os 
if not parallel:
    os.environ["POLARS_MAX_THREADS"] = "1"

import polars as pl
import pandas as pd
import torch as np
from scipy.special import expit
import time
import csv
from threadpoolctl import threadpool_limits


def simplify_col(table, column):
    vals = table[column].unique()
    val_to_id = dict()
    for i, val in enumerate(vals):
        val_to_id[val] = i
    table[column] = [val_to_id[v] for v in table[column]]

def one_hot_encode_col(table, column):
    one_hot = pd.get_dummies(table[column], dtype=float)
    table = table.drop(column,axis = 1)
    table = table.join(one_hot)
    return table

def one_hot_encode_cols(table, columns):
    for column in columns:
        table = one_hot_encode_col(table, column)
    return table

def star_join(lineitem, orders, customer, supplier, part):
    x = lineitem.join(orders, on="OrderKey")
    y = x.join(customer, on="CustomerKey")
    z = y.join(supplier, on="SuppKey", suffix="y")
    u = z.join(part, on="PartKey")
    return u

def lr_1(lineitem, orders, customer, supplier, part, theta):
    X = np.from_numpy(star_join(lineitem, orders, customer, supplier, part).to_numpy())
    return np.matmul(X, theta)

def log_1(lineitem, orders, customer, supplier, part, theta):
    X = np.from_numpy(star_join(lineitem, orders, customer, supplier, part).to_numpy())
    return expit(np.matmul(X, theta))

def cov_1(lineitem, orders, customer, supplier, part):
    X = np.from_numpy(star_join(lineitem, orders, customer, supplier, part).to_numpy())
    return np.matmul(X.T, X)

def nn_1(lineitem, orders, customer, supplier, part, W1, W2, W3):
    X = np.from_numpy(star_join(lineitem, orders, customer, supplier, part).to_numpy())
    h1 = np.maximum(np.scalar_tensor(0), np.matmul(X, W1))
    h2 = np.maximum(np.scalar_tensor(0), np.matmul(h1, W2))
    return expit(np.matmul(h2, W3))

def self_join(lineitem, part, supplier):
    return lineitem.join(lineitem, on="PartKey", suffix="_x").join(part, on="PartKey").join(supplier, left_on="SuppKey_x", right_on="SuppKey", suffix="_1").join(supplier, left_on="SuppKey", right_on="SuppKey", suffix="_2")

def lr_2(lineitem, part, supplier, theta):
    X = np.from_numpy(self_join(lineitem, part, supplier).to_numpy())
    return np.matmul(X, theta)

def log_2(lineitem, part, supplier, theta):
    X = np.from_numpy(self_join(lineitem, part, supplier).to_numpy())
    return expit(np.matmul(X, theta))

def cov_2(lineitem, part, supplier):
    X = np.from_numpy(self_join(lineitem, part, supplier).to_numpy())
    return np.matmul(X.T, X)

def nn_2(lineitem, part, supplier, W1, W2, W3):
    X = np.from_numpy(self_join(lineitem, part, supplier).to_numpy())
    h1 = np.maximum(np.scalar_tensor(0), np.matmul(X, W1))
    h2 = np.maximum(np.scalar_tensor(0), np.matmul(h1, W2))
    return expit(np.matmul(h2, W3))


def main():
    num_cores = 24 if parallel else 1
    with threadpool_limits(limits=num_cores):
        customer = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH_5/customer.tbl", sep="|", names=["CustomerKey", "Name", "Address", "NationKey", "Phone", "AcctBal", "MktSegment", "Comment", "Col9"])
        lineitem = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH_5/lineitem.tbl", sep="|", names=["OrderKey", "PartKey", "SuppKey", "LineNumber", "Quantity", "ExtendedPrice", "Discount", "Tax", "ReturnFlag", "LineStatus", "ShipDate", "CommitDate", "ReceiptDate", "ShipInstruct", "ShipMode", "Comment", "Col9"])
        lineitem["LineItemKey"] = range(1, len(lineitem)+1) 
        orders = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH_5/orders.tbl", sep="|", names=["OrderKey", "CustomerKey", "OrderStatus", "TotalPrice", "OrderDate", "OrderPriority", "Clerk", "ShipPriority", "Comment", "Extra"])
        partsupp = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH_5/partsupp.tbl", sep="|", names=["PartKey", "SuppKey", "AvailQty", "SupplyCost", "Comment", "Col9"])
        part = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH_5/part.tbl", sep="|", names=["PartKey", "Name", "MFGR", "Brand", "Type", "Size", "Container", "RetailPrice", "Comment", "Col9"])
        supplier = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH_5/supplier.tbl", sep="|", names=["SuppKey", "Name", "Address", "NationKey", "Phone", "AcctBal", "Comment", "Col9"])

        simplify_col(lineitem, "OrderKey")
        simplify_col(lineitem, "PartKey")
        simplify_col(lineitem, "SuppKey")
        simplify_col(orders, "OrderKey")
        simplify_col(orders, "CustomerKey")
        simplify_col(customer, "CustomerKey")
        simplify_col(part, "PartKey")
        simplify_col(supplier, "SuppKey")

        lineitem = pl.from_pandas(lineitem[["LineItemKey", "OrderKey", "PartKey", "SuppKey"]])
        orders = orders[["OrderKey", "CustomerKey", "OrderStatus", "TotalPrice", "OrderPriority", "ShipPriority"]]
        orders = pl.from_pandas(one_hot_encode_cols(orders, ["OrderStatus", "OrderPriority", "ShipPriority"]))
        customer = customer[["CustomerKey", "NationKey", "AcctBal", "MktSegment"]]
        customer = pl.from_pandas(one_hot_encode_cols(customer, ["NationKey", "MktSegment"]))
        partsupp = pl.from_pandas(partsupp[["PartKey", "SuppKey"]])
        supplier = supplier[["SuppKey", "NationKey", "AcctBal"]]
        supplier = pl.from_pandas(one_hot_encode_cols(supplier, ["NationKey"]))
        part = part[["PartKey", "MFGR", "Brand", "Size", "Container", "RetailPrice"]]
        part = pl.from_pandas(one_hot_encode_cols(part, ["MFGR", "Brand", "Container"]))
        print(lineitem)
        theta = np.rand(144, dtype=float)
        W1 = np.rand(144, 25, dtype=float)
        W2 = np.rand(25, 25, dtype=float)
        W3 = np.rand(25, dtype=float)

        np_lr_time = 0
        np_log_time = 0
        np_cov_time = 0
        np_nn_time = 0
        n = 3
        for iter in range(0, n + 1):
            if iter == 1:
                np_lr_time = 0
                np_log_time = 0
                np_cov_time = 0
                np_nn_time = 0
            
            np_start = time.time()
            lr_1(lineitem, orders, customer, supplier, part, theta)
            np_end = time.time()
            np_lr_time = np_lr_time + np_end-np_start
            print(np_end-np_start)

            np_start = time.time()
            log_1(lineitem, orders, customer, supplier, part, theta)
            np_end = time.time()
            np_log_time = np_log_time + np_end-np_start
            print(np_end-np_start)

            np_start = time.time()
            cov_1(lineitem, orders, customer, supplier, part)
            np_end = time.time()
            np_cov_time = np_cov_time + np_end-np_start
            print(np_end-np_start)

            np_start = time.time()
            nn_1(lineitem, orders, customer, supplier, part, W1, W2, W3)
            np_end = time.time()
            np_nn_time = np_nn_time + np_end-np_start
            print(np_end-np_start)
        
        np_lr_time = np_lr_time / n
        np_log_time = np_log_time / n
        np_cov_time = np_cov_time / n
        np_nn_time = np_nn_time / n



        customer = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH/customer.tbl", sep="|", names=["CustomerKey", "Name", "Address", "NationKey", "Phone", "AcctBal", "MktSegment", "Comment", "Col9"])
        lineitem = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH/lineitem.tbl", sep="|", names=["OrderKey", "PartKey", "SuppKey", "LineNumber", "Quantity", "ExtendedPrice", "Discount", "Tax", "ReturnFlag", "LineStatus", "ShipDate", "CommitDate", "ReceiptDate", "ShipInstruct", "ShipMode", "Comment", "Col9"])
        lineitem["LineItemKey"] = range(1, len(lineitem)+1) 
        orders = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH/orders.tbl", sep="|", names=["OrderKey", "CustomerKey", "OrderStatus", "TotalPrice", "OrderDate", "OrderPriority", "Clerk", "ShipPriority", "Comment", "Extra"])
        partsupp = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH/partsupp.tbl", sep="|", names=["PartKey", "SuppKey", "AvailQty", "SupplyCost", "Comment", "Col9"])
        part = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH/part.tbl", sep="|", names=["PartKey", "Name", "MFGR", "Brand", "Type", "Size", "Container", "RetailPrice", "Comment", "Col9"])
        supplier = pd.read_csv("/local1/kdeeds/Galley.jl/Experiments/Data/TPCH/supplier.tbl", sep="|", names=["SuppKey", "Name", "Address", "NationKey", "Phone", "AcctBal", "Comment", "Col9"])

        simplify_col(lineitem, "OrderKey")
        simplify_col(lineitem, "PartKey")
        simplify_col(lineitem, "SuppKey")
        simplify_col(orders, "OrderKey")
        simplify_col(orders, "CustomerKey")
        simplify_col(customer, "CustomerKey")
        simplify_col(part, "PartKey")
        simplify_col(supplier, "SuppKey")

        lineitem = pl.from_pandas(lineitem[["LineItemKey", "OrderKey", "PartKey", "SuppKey"]])
        orders = orders[["OrderKey", "CustomerKey", "OrderStatus", "TotalPrice", "OrderPriority", "ShipPriority"]]
        orders = pl.from_pandas(one_hot_encode_cols(orders, ["OrderStatus", "OrderPriority", "ShipPriority"]))
        customer = customer[["CustomerKey", "NationKey", "AcctBal", "MktSegment"]]
        customer = pl.from_pandas(one_hot_encode_cols(customer, ["NationKey", "MktSegment"]))
        partsupp = pl.from_pandas(partsupp[["PartKey", "SuppKey"]])
        supplier = supplier[["SuppKey", "NationKey", "AcctBal"]]
        supplier = pl.from_pandas(one_hot_encode_cols(supplier, ["NationKey"]))
        part = part[["PartKey", "MFGR", "Brand", "Size", "Container", "RetailPrice"]]
        part = pl.from_pandas(one_hot_encode_cols(part, ["MFGR", "Brand", "Container"]))
        theta_sj = np.rand(131, dtype=float)
        W1_sj = np.rand(131, 25, dtype=float)
        W2_sj = np.rand(25, 25, dtype=float)
        W3_sj = np.rand(25, dtype=float)

        np_lr_time2 = 0
        np_log_time2 = 0
        np_cov_time2 = 0
        np_nn_time2 = 0
        for iter in range(0, n + 1):
            if iter == 1:
                np_lr_time2 = 0
                np_log_time2 = 0
                np_cov_time2 = 0
                np_nn_time2 = 0
            np_start = time.time()
            lr_2(lineitem, part, supplier, theta_sj)
            np_end = time.time()
            np_lr_time2 = np_lr_time2 + np_end-np_start
            print(np_end-np_start)

            np_start = time.time()
            log_2(lineitem, part, supplier, theta_sj)
            np_end = time.time()
            np_log_time2 = np_log_time2 + np_end-np_start
            print(np_end-np_start)

            np_start = time.time()
            cov_2(lineitem, part, supplier)
            np_end = time.time()
            np_cov_time2 = np_cov_time2 + np_end-np_start
            print(np_end-np_start)

            np_start = time.time()
            nn_2(lineitem, part, supplier, W1_sj, W2_sj, W3_sj)
            np_end = time.time()
            np_nn_time2 = np_nn_time2 + np_end-np_start
            print(np_end-np_start)


        np_lr_time2 = np_lr_time2 / n
        np_log_time2 = np_log_time2 / n
        np_cov_time2 = np_cov_time2 / n
        np_nn_time2 = np_nn_time2 / n
        method_name = "Polars+PyTorch (24 Core)" if parallel else "Polars+PyTorch (1 Core)"
        data = [["Method", "Algorithm", "ExecuteTime", "OptTime"],
                [method_name, "Linear Regression (SQ)", np_lr_time, 0],
                [method_name, "Logistic Regression (SQ)", np_log_time, 0],
                [method_name, "Covariance (SQ)", np_cov_time, 0],
                [method_name, "Neural Network (SQ)", np_nn_time, 0],
                [method_name, "Linear Regression (SJ)", np_lr_time2, 0],
                [method_name, "Logistic Regression (SJ)", np_log_time2, 0],
                [method_name, "Covariance (SJ)", np_cov_time2, 0],
                [method_name, "Neural Network (SJ)", np_nn_time2, 0]]
        filename = "/local1/kdeeds/Galley.jl/Experiments/Results/tpch_inference_polars_parallel.csv" if parallel else "/local1/kdeeds/Galley.jl/Experiments/Results/tpch_inference_polars_serial.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print("Numpy LR", np_lr_time)
        print("Numpy Log", np_log_time)
        print("Numpy Cov", np_cov_time)
        print("Numpy NN", np_nn_time)
        print("Numpy LR2", np_lr_time2)
        print("Numpy Log2", np_log_time2)
        print("Numpy Cov2", np_cov_time2)
        print("Numpy NN2", np_nn_time2)

main()
