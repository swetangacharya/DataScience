**LU Decoposition**

It is a matrix factorization method where it factors a square matrix into Lower and Upper triangular matrix L and U  such that A=L U
The primary importance of LU decomposition is to solve the linear systems AX=Y, perticularly, when solving same A but different Y. The matrix L result from clearing all the values above the main diagonal via Gaussian Elimination.
The matrix U reflects what row operations have occurred in the course of building L.Simplifying U and applying L to Y requires less work than solving A from scratch.
Also, LU decomposition help us to determine the determinant of Matrix. The product of Diagonal of U is a determinant.
Look at the grayscale image of 64x64 pixels. we'll use scipy and numpy to decompose it in L and U and the permutation matrix P.

![image](https://github.com/user-attachments/assets/7dddd88c-c221-40da-a6bd-f0f0cb9b3f68)

```
from PIL import Image
import numpy as np
from scipy.linalg import lu
from matplotlib import pyplot as plt

img=Image.open('gray64x64.png')
A=np.array(img)
P,L,U=lu(A)

#plt.imshow(L,interpolation='nearest')
#plt.imshow(U,interpolation='nearest')
#plt.imshow(P@L@U,interpolation='nearest')
plt.show()
```
This is How it looks when line 16 is uncommendted.
<img width="297" alt="image" src="https://github.com/user-attachments/assets/a24a0f06-4cd8-4cd8-9004-e5eba80bc8ed" />

This is How it looks when line 17 is uncommendted.
<img width="295" alt="image" src="https://github.com/user-attachments/assets/402c1304-d8e0-46f4-bd76-15321db66c52" />

**You can see above that without appliying permutation, we've bottom lines shifted to top.**
This is How it looks when line 18 is uncommendted.
<img width="303" alt="image" src="https://github.com/user-attachments/assets/857054a3-65d5-47d4-bd89-94e648d77562" />

