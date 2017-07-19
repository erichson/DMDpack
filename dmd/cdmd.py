"""

Compressed Dynamic Mode Decomposition (DMD) python function.

"""

from __future__ import division
import numpy as np
import scipy as sci
from scipy import linalg
import scipy.sparse.linalg as scislin

from numpy.testing import assert_raises
 
from rsvd import rsvd
from hfun import *
    

def cdmd(A, dt = 1, k=None, p=10, sdist='sparse', sf=0.9, modes='exact',
         return_amplitudes=False, return_vandermonde=False, order=True, trace=True):
    """
    Compressed Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `a` in space and time. The matrix `a` is 
    decomposed as `a = FBV`, where the columns of `F` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes stored in the diagonal 
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.


    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    
    dt : scalar or array_like  
        Factor specifying the time difference between the observations.  
    
    k : int
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    
    p : int, optional
        Oversampling paramater.         
    
    sdist : str `{'unif', 'norm', 'sparse', 'spixel'}`  
        Specify the distribution of the sensing matrix `S`. 
    
    sf : int, optional
        `sf` sets the sparsity factor for the `sparse` sdist, i.e. `sf=0.9` means
        `90%` of the sensing matrix `S` entries are zero.
    
    modes : str `{'exact', 'exact_scaled'}`
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
        
        'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.
    
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.     
    
    order :  bool `{True, False}`
        True: return modes sorted.


    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    
    b : array_like
        1-D array containing the amplitudes of length `min(n-1, k)`.
    
    V : array_like
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.


    Notes
    -----
    
    
    References
    ----------
    S. L. Brunton, et al.
    "Compressed Sensing and Dynamic Mode Decomposition" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.5186>`_).    
    
    J. H. Tu, et al.
    "On Dynamic Mode Decomposition: Theory and Applications" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).    

    
    Examples
    --------



    
    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************
 
    #Shape of A
    m, n = A.shape   
    dat_type =  A.dtype
    if  dat_type == np.float32: 
        isreal = True
        real_type = np.float32
        fT = rT
    elif dat_type == np.float64: 
        isreal = True
        real_type = np.float64 
        fT = rT
    elif dat_type == np.complex64:
        isreal = False 
        real_type = np.float32
        fT = cT
    elif dat_type == np.complex128:
        isreal = False 
        real_type = np.float64
        fT = cT
    else:
        raise ValueError('A.dtype is not supported')


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compress
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Ac = A
    if sdist=='unif':   
            S = np.array( np.random.uniform( -1 , 1 , size=( k+p, m ) ) , dtype = dat_type ) 
            if isreal==False: 
                S += 1j * np.array( np.random.uniform(-1 , 1 , size=( k+p, m ) ) , dtype = real_type )
            #Compress input matrix
            Ac = S.dot(A)    
            del(S)

    elif sdist=='norm':   
            S = np.array( np.random.standard_normal( size=( k+p, m ) ) , dtype = dat_type ) 
            if isreal==False: 
                S += 1j * np.array( np.random.standard_normal( size=( k+p, m ) ) , dtype = real_type )     
            #Compress input matrix
            Ac = S.dot(A)    
            del(S)                
            
    elif sdist=='sparse':   
            density = 1-sf
            S = sci.sparse.rand(k+p, m, density=density, format='coo', random_state=None)
            #if isreal==False: 
            #    S.data += 1j * np.random.uniform(0 , 1 , size=( len(S.data) ) )  
            if trace==True: print "Sparse: %f" %sf + " zeros "
            S.data *=2          
            S.data -=1            
            S = S.tocsr()
            #Compress input matrix
            Ac = S.dot(A)    
            del(S)
        
    elif sdist=='spixel':
            rrows = np.random.choice( np.arange(m), size=k+p, replace=False, p=None)
            Ac =   np.array( A[ rrows , : ] , dtype = dat_type )
            
    else: 
            raise ValueError('Sampling distribution is not supported.') 
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    X = Ac[ : , xrange( 0 , n-1 ) ] #pointer
    Y = Ac[ : , xrange( 1 , n ) ] #pointer   
     
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    if k != None:
         U, s, Vh = sci.linalg.svd( X ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=False,
                                  check_finite=True)
         U = U[ : , xrange(k) ]
         s = s[ xrange(k) ]
         Vh = Vh[ xrange(k) , : ]

    else:
         U, s, Vh = sci.linalg.svd( X ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=False,
                                  check_finite=True)
    #EndIf    
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Solve the LS problem to find estimate for M using the pseudo-inverse    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #real: M = U.T * Y * Vt.T * S**-1
    #complex: M = U.H * Y * Vt.H * S**-1
    #Let G = Y * Vt.H * S**-1, hence M = M * G
    Vscaled = fT(Vh) * s**-1
    G = np.dot( Y , Vscaled )
    M = np.dot( fT(U) , G ) 
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    l, W = sci.linalg.eig( M , right=True, overwrite_a=True )    
 
    omega = np.log(l) / dt
       
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Order
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if order==True: 
        sort_idx = np.argsort(np.abs(omega)) 
        W = W[  :, sort_idx ]
        l = l[ sort_idx ] 
        omega = omega[ sort_idx ]  
          
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if modes=='exact': 
        F = np.dot( A[ : , xrange( 1 , n ) ] , np.dot( Vscaled , W ) )   
    elif modes=='exact_scaled':  
        F = np.dot( A[ : , xrange( 1 , n ) ] , np.dot( Vscaled , W ) ) * ( 1/l )
    else: 
        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')
    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if return_amplitudes==True:   
        b , _ , _ , _ = sci.linalg.lstsq( F , A[ : , 0 ])


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True: 
        V = np.fliplr(np.vander( l , N =  n ))     
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
    if return_amplitudes==True and return_vandermonde==True:
        return F, b, V, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F, b, omega
    else:
        return F , omega   
  
    #**************************************************************************   
    #End cDMD
    #**************************************************************************  

