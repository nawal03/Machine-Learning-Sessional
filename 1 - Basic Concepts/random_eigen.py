import numpy as np
import random

n = int(input("Enter the dimension: "))

matrix = np.zeros((n, n))

for i in range(0,n):
    abs_sum=0
    for j in range(0,n):
        if i!=j:
            matrix[i,j]=random.randint(-n*n,n*n)
            abs_sum+=abs(matrix[i,j])
    matrix[i,i]=random.randint(abs_sum+1,abs_sum+100)

print("The invertible matrix is: \n", matrix)

eigen_values, eigen_vectors = np.linalg.eig(matrix)
print("eigen values: ", eigen_values)
print("eigen vectors: \n", eigen_vectors)

diagonal_matrix = np.diag(eigen_values)
reconstructed_matrix = eigen_vectors @ diagonal_matrix @ np.linalg.inv(eigen_vectors)

# Reconstruction correctness checking
correct = np.allclose(matrix, reconstructed_matrix)

if correct:
    print("Reconstruction is correct.")
else:
    print("Reconstruction is incorrect.")