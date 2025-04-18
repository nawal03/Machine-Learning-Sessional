import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U, S, Vt = np.linalg.svd(A)

    Sk = S[:k]
    Dk = np.diag(Sk)
    Uk = U[:, :k]
    Vtk = Vt[:k, :]

    Ak = Uk @ Dk @ Vtk
    return Ak


# Read the image
image = cv2.imread('image.jpg')

# Check if the image was read successfully
if image is not None:
    # Resize the image to a lower dimension
    width = 500
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))

    # Convert the resized image to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Plot k-rank approximations
    plot_rows = 3
    plot_cols = 4
    k=[1, 5, 10, 20, 30, 40, 45, 50, 100, 200, 400, 500]
    plt.figure(figsize=(18, 9))

    for i in range(0, len(k)):
        approximated_matrix = low_rank_approximation(grayscale_image, k[i])
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.imshow(approximated_matrix, cmap='gray')
        plt.title(f'n_components = {k[i]}',fontsize=9)
        plt.tick_params(axis='both', which='both', labelsize=7)
    plt.tight_layout()
    plt.show()
    print('lowest k is 30')
else:
    print("Error: Image not found or unable to read.")
