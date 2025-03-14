import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    pass

def distance_l2(X, Y):
    pass

def distance_dot(X, Y):
    pass

def distance_manhattan(X, Y):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def distance_kernel(X, Y, D, metric="L2"):
    """
    Computes L2 (Euclidean distance) or Cosine similarity.

    Parameters:
    - X: Data points (N, D)
    - Y: Centroids (K, D)
    - D: Dimension of vectors
    - metric: "L2" (Euclidean distance) or "cosine" (Cosine similarity)

    Returns:
    - Distance matrix (N, K)
    """
    if metric == "L2":
        return torch.cdist(X, Y, p=2)  # Euclidean distance
    elif metric == "cosine":
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
        return 1 - torch.mm(X_norm, Y_norm.T)  # 1 - Cosine similarity
    else:
        raise ValueError("metric must be 'L2' or 'cosine'")

def our_kmeans(N, D, A, K, metric="L2", max_iters=100, tol=1e-4):
    """
    K-Means clustering algorithm, supporting L2 and Cosine similarity metrics.
    This function strictly follows the given code template.

    Parameters:
    - N: Number of data points
    - D: Dimension of each data point
    - A: Dataset (numpy array)
    - K: Number of clusters
    - metric: "L2" (Euclidean distance) or "cosine" (Cosine similarity)
    - max_iters: Maximum iterations for convergence
    - tol: Convergence threshold

    Returns:
    - labels: Cluster ID for each data point (PyTorch Tensor)
    """

    # Enforce GPU usage; raise an error if GPU is not available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Please run this code on a GPU-enabled system.")

    device = "cuda"
    print(f"Running on: {device}, using {metric} as the distance metric")

    # Convert dataset to PyTorch Tensor and move to GPU
    A = torch.tensor(A, dtype=torch.float32).to(device)

    # Initialize K random centroids
    centroids = A[torch.randperm(N)[:K]].clone()

    for i in range(max_iters):
        if metric == "L2":
            # Compute L2 (Euclidean) distances
            distances = torch.cdist(A, centroids, p=2)
        elif metric == "cosine":
            # Compute Cosine similarity (1 - Cosine Similarity used as "distance")
            A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
            centroids_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
            distances = 1 - torch.mm(A_norm, centroids_norm.T)
        else:
            raise ValueError("metric must be 'L2' or 'cosine'")

        # Assign each point to the nearest cluster
        labels = torch.argmin(distances, dim=1)

        # Compute new centroids
        new_centroids = torch.zeros_like(centroids, device=device)
        for k in range(K):
            cluster_points = A[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(dim=0)

        # Compute centroid shift
        shift = torch.norm(new_centroids - centroids, p=2, dim=1).max().item()

        # Update centroids
        centroids = new_centroids

        # Check for convergence
        if shift < tol:
            print(f"K-Means converged at iteration {i+1}")
            break

    return labels.cpu()  # Return cluster IDs

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
