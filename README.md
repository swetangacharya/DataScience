**LU Decoposition**

It is a matrix factorization method where it factors a square matrix into Lower and Upper triangular matrix L and U  such that A=L U
The primary importance of LU decomposition is to solve the linear systems AX=Y, perticularly, when solving same A but different Y. The matrix L result from clearing all the values above the main diagonal via Gaussian Elimination.
The matrix U reflects what row operations have occurred in the course of building L.Simplifying U and applying L to Y requires less work than solving A from scratch.
Also, LU decomposition help us to determine the determinant of Matrix. The product of Diagonal of U is a determinant.
Look at the grayscale image of 64x64 pixels. we'll use scipy and numpy to decompose it in L and U and the permutation matrix P.

![image](https://github.com/user-attachments/assets/7dddd88c-c221-40da-a6bd-f0f0cb9b3f68)

```
img=Image.open('gray64x64.png')
A=np.array(img)
P,L,U=lu(A)

#plt.imshow(L,interpolation='nearest')
#plt.imshow(U,interpolation='nearest')
#plt.imshow(P@L@U,interpolation='nearest')
plt.show()
```
This is How it looks when line 16 is uncommendted.
![image](https://github.com/user-attachments/assets/57e38093-2a8d-4a26-b43b-c945817059e1)

This is How it looks when line 17 is uncommendted.
![image](https://github.com/user-attachments/assets/ed567cf7-78c0-4a68-bbce-cd198533e2f9)

> You can see above that without appliying permutation, we've bottom lines shifted to top.
This is How it looks when line 18 is uncommendted.
![image](https://github.com/user-attachments/assets/68fa5912-6a69-4ccf-9cc4-b85e5638cfcc)
