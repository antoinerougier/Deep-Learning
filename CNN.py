"""
Convolution 2D from Scratch
============================
Entrées :
  - matrice d'entrée  (H × W)
  - matrice filtre    (Fh × Fw)
  - stride            (déplacement)
  - padding           (optionnel, défaut 0)

Sorties :
  - feature map (carte de caractéristiques)
"""


def convolve2d(matrix, kernel, stride=1):
    """
    Convolution 2D valide.

    Paramètres
    ----------
    matrix  : liste de listes de floats/ints  (H × W)
    kernel  : liste de listes de floats/ints  (Fh × Fw)
    stride  : int — pas de déplacement (défaut 1)
    padding : int — zéros ajoutés autour de la matrice (défaut 0)

    Retourne
    --------
    feature_map : liste de listes de floats
    """

    H  = len(matrix)
    W  = len(matrix[0])
    Fh = len(kernel)
    Fw = len(kernel[0])

    padded = matrix
    H_p, W_p = H, W

    out_H = (H_p - Fh) // stride + 1
    out_W = (W_p - Fw) // stride + 1

    if out_H <= 0 or out_W <= 0:
        raise ValueError(
            f"Le filtre ({Fh}x{Fw}) est trop grand pour l'entree "
            f"({H_p}x{W_p}) avec stride={stride}."
        )

    feature_map = [[0.0] * out_W for _ in range(out_H)]

    for i in range(out_H):
        for j in range(out_W):
            row_start = i * stride
            col_start = j * stride
            total = 0.0
            for fi in range(Fh):
                for fj in range(Fw):
                    total += padded[row_start + fi][col_start + fj] * kernel[fi][fj]
            feature_map[i][j] = total

    return feature_map


def print_matrix(m, title='', fmt='{:7.2f}'):
    if title:
        print(f"\n{title}")
        print("-" * (len(title) + 2))
    for row in m:
        print(" ".join(fmt.format(v) for v in row))


if __name__ == "__main__":

    print("=" * 55)
    print("  EXEMPLE 1 - Filtre Sobel horizontal (stride=1)")
    print("=" * 55)

    image = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    sobel_h = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ]
    fm = convolve2d(image, sobel_h, stride=1)
    print_matrix(image,   "Entree (7x7)")
    print_matrix(sobel_h, "Filtre Sobel horizontal (3x3)")
    print_matrix(fm,      "Feature map (5x5)")

    print("\n" + "=" * 55)
    print("  EXEMPLE 2 - Filtre moyenneur avec stride=2")
    print("=" * 55)

    matrix = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9,10,11,12],
        [1, 3, 5, 7, 9,11],
        [2, 4, 6, 8,10,12],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
    mean_filter = [
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ]
    fm2 = convolve2d(matrix, mean_filter, stride=2)
    print_matrix(matrix,      "Entree (6x6)")
    print_matrix(mean_filter, "Filtre moyenneur (3x3)")
    print_matrix(fm2,         f"Feature map ({len(fm2)}x{len(fm2[0])}) - stride=2")

    # Exemple 3 : padding same
    print("\n" + "=" * 55)
    print("  EXEMPLE 3 - Padding=1 (taille conservee)")
    print("=" * 55)

    small = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    sharpen = [
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ]
    fm3 = convolve2d(small, sharpen, stride=1)
    print_matrix(small,   "Entree (3x3)")
    print_matrix(sharpen, "Filtre sharpen (3x3)")
    print_matrix(fm3,     f"Feature map ({len(fm3)}x{len(fm3[0])}) - meme taille")

    print("\n" + "=" * 55)
    print("  FORMULES")
    print("=" * 55)
    print("  out_H = floor((H + 2P - Fh) / stride) + 1")
    print("  out_W = floor((W + 2P - Fw) / stride) + 1")
    print("  Same convolution : P = (F - 1) / 2  avec stride=1")
    print("=" * 55)