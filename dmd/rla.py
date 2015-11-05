"""

Dynamic Mode Decomposition (DMD) python function.

"""

from __future__ import division
import numpy as np
import scipy as sci
import scipy.sparse.linalg as scislin
from numpy.testing import assert_raises
    

def rsvd(A, k=None, p=0, q=0, method='standard', sdist='unif'):
    """
    Randomized Singular Value Decomposition.
    
    Randomized algorithm for computing the approximate low-rank singular value 
    decomposition of a rectangular (m, n) matrix `a` with target rank `k << n`. 
    The input matrix a is factored as `a = U * diag(s) * Vt`. The right singluar 
    vectors are the columns of the real or complex unitary matrix `U`. The left 
    singular vectors are the columns of the real or complex unitary matrix `V`. 
    The singular values `s` are non-negative and real numbers.

    The paramter `p` is a oversampling parameter to improve the approximation. 
    A value between 2 and 10 is recommended.
    
    The paramter `q` specifies the number of normlized power iterations
    (subspace iterations) to reduce the approximation error. This is recommended 
    if the the singular values decay slowly and in practice 1 or 2 iterations 
    achive good results. However, computing power iterations is increasing the
    computational time. 
    
    If k > (n/1.5), partial SVD or trancated SVD might be faster. 
    
    
    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int
        `k` is the target rank of the low-rank decomposition, k << min(m,n). 
    p : int
        `p` sets the oversampling parameter (default k=0).
    q : int
        `q` sets the number of power iterations (default=0).
    method : str `{'standard', 'fast'}`
        'standard' : Standard algorithm as described in [1, 2].
        'fast' : Version II algorithm as described in [2].                 
    sdist : str `{'unif', 'punif', 'norm', 'sparse', 'vsparse'}`
    
    
    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, k)`.
    s : array_like
        Singular values, 1-d array of length `k`.
    Vh : array_like
        Left singular values, array of shape `(k, n)`.


    Notes
    -----   
    Input matrix of shape `(m, n)`, where `n>m` is not supported yet.


    References
    ----------
    N. Halko, P. Martinsson, and J. Tropp.
    "Finding structure with randomness: probabilistic
    algorithms for constructing approximate matrix
    decompositions" (2009).
    (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    
    S. Voronin and P.Martinsson. 
    "RSVDPACK: Subroutines for computing partial singular value 
    decompositions via randomized sampling on single core, multi core, 
    and GPU architectures" (2015).
    (available at `arXiv <http://arxiv.org/abs/1502.05366>`_).


    Examples
    --------


    
    """
    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************
    
    # Shape of input matrix 
    m , n = A.shape   
    dat_type =  A.dtype   

    if  dat_type == np.float32: 
        isreal = True
        real_type = np.float32
    elif dat_type == np.float64: 
        isreal = True
        real_type = np.float64  
    elif dat_type == np.complex64:
        isreal = False 
        real_type = np.float32
    elif dat_type == np.complex128:
        isreal = False 
        real_type = np.float64
    else:
        return "A.dtype is not supported"
    
    if m < n:
        A = A.conj().T
        m , n = A.shape 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random sampling matrix O
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='unif':   
        O = np.array( np.random.uniform( -1 , 1 , size=( n, k+p ) ) , dtype = dat_type ) 
        if isreal==False: 
            O += 1j * np.array( np.random.uniform(-1 , 1 , size=( n, k+p  ) ) , dtype = dat_type )

    if sdist=='punif':   
        O = np.array( np.random.uniform( 0 , 1 , size=( n, k+p  ) ) , dtype = dat_type ) 
        if isreal==False: 
            O += 1j * np.array( np.random.uniform(0 , 1 , size=( n, k+p  ) ) , dtype = dat_type )
      
    elif sdist=='norm':   
        O = np.array( np.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type ) 
        if isreal==False: 
            O += 1j * np.array( np.random.standard_normal( size=( n, k+p  ) ) , dtype = dat_type )     
 
    elif sdist=='sobol':   
        O = np.array( i4_sobol_generate( n, k+p  ) , dtype = dat_type).toarray() 
        if isreal==False: 
            O += 1j * np.array( i4_sobol_generate( n, k+p  ) , dtype = dat_type).toarray() 
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * O
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Y = A.dot(O)

    del(O)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Orthogonalize Y using economic QR decomposition: Y=QR
    #If q > 0 perfrom q subspace iterations
    #Note: check_finite=False may give a performance gain
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    
    if q > 0:
        for i in np.arange( 1, q+1 ):
            if( (2*i-2) % q == 0 ):
                Y , _ = sci.linalg.qr( Y , mode='economic', check_finite=True )
            
            Z = np.dot( A.conj().T , Y )
            
            if( (2*i-1) % q == 0 ):
                Z , _ = sci.linalg.qr( Z , mode='economic', check_finite=True)
       
            Y = np.dot( A , Z )             
        #End for
     #End if       
        
    Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=True ) 
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Project the data matrix a into a lower dimensional subspace
    #B = Q.T * A 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    B = np.dot( Q.conj().T , A )   

    if method == 'standard':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Singular Value Decomposition
        #Note: B = U" * S * Vt
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
        #Compute SVD
        U , s , Vh = sci.linalg.svd( B ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=True,
                                  check_finite=True)
         
        #Recover right singular vectors
        U = np.dot( Q , U)

        #Return Trunc
        return ( U[ : , range(k) ] , s[ range(k) ] , Vh[ range(k) , : ] ) 
    
    if method == 'fast':
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Orthogonalize B.T using reduced QR decomposition: B.T = Q" * R"
        #Note: reduced QR returns Q and R, and destroys B_gpu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Compute QR
        Qstar, Rstar = sci.linalg.qr( B.conj().T,  mode='economic', check_finite=True) 

        #Compute right singular vectors
        Ustar , s , Vstar = sci.linalg.svd( Rstar , compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=True,
                                  check_finite=True)
 
        U =  np.dot( Q , Vstar.conj().T )   
        V =  np.dot( Qstar , Ustar ) 
        
        
        #Return Trunc
        return ( U[ : , range(k) ] , s[ range(k) ] , V[ : , range(k) ].conj().T ) 
        
    #**************************************************************************   
    #End rsvd
    #**************************************************************************       

     
if __name__ == "__main__":
	print "Imported."