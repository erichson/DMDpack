DMDpack: Implementations of standard and fast (randomized / compressed) algorithms 
for computing full and low rank dynamic mode decompositions of a matrix.

Dynamic Mode Decomposition (DMD) is a data processing algorithm which
allows to decompose a matrix `a` in space and time.
The matrix `a` is decomposed as `a = FBV`, where the columns of `F`
contain the dynamic modes. The modes are ordered corresponding
to the amplitudes stored in the diagonal matrix `B`. `V` is a Vandermonde
matrix describing the temporal evolution.