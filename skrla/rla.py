from __future__ import division
import numpy as np
import scipy as sci
import scipy.sparse.linalg as scislin
from numpy.testing import assert_raises
#**************************************************************************   
#************************************************************************** 


def dmd(A, k=None, p=5, q=2, modes='standard', return_amplitudes=True,
        return_vandermonde=True, svd='rand', rsvd_type='fast', order=True):
    """
    Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `a` in space and time.
    The matrix `a` is decomposed as `a = FBV`, where the columns of `F`
    contain the dynamic modes. The modes are ordered corresponding
    to the amplitudes stored in the diagonal matrix `B`. `V` is a Vandermonde
    matrix describing the temporal evolution.

    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    p : int, optional
        `p` sets the oversampling parameter for rSVD (default k=5).
    q : int, optional
        `q` sets the number of power iterations for rSVD (default=1).
    modes : str `{'standard', 'exact', 'exact_scaled'}`
        'standard' : uses the standard definition to compute the dynamic modes,
                    `F = U * W`.
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
        'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.
    svd : str `{'rand', 'partial', 'trancated'}`
        'rand' : uses randomized singular value decomposition (default). 
        'partial' : uses partial singular value decomposition.
        'trancated' : uses trancated singular value decomposition.
    rsvd_type : str `{'standard', 'fast'}`
        'standard' : (default) Standard algorithm as described in [1, 2]. 
        'fast' : Version II algorithm as described in [2].       
    order :  bool `{True, False}`
        True: return modes sorted according to the amplitudes

    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    b_gpu : array_like
        1-D array containing the amplitudes of length `min(n-1, k)`.
    V : array_like
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    Notes
    -----
    

    References
    ----------
    J. H. Tu, et al.
    "On dynamic mode decomposition: theory and applications."
    arXiv preprint arXiv:1312.0041 (2013).
    
    N. B. Erichson and C. Donovan.
    "Randomized Low-Rank Dynamic Mode Decomposition for Motion Detection"
    Under Review.    


    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************
 
    #Shape of D
    m, n = A.shape   

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = A[ : , range( 0 , n-1 ) ] #pointer
    Y = A[ : , range( 1 , n ) ] #pointer   
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    if k != None:
        if svd=="rand":
            U, s, Vh = rsvd( X, k=k , p=p , q=q , method=rsvd_type )  
        
        elif svd=="csvd":
            U, s, Vh = cSVD( X , k=k, q=q)  
        
        elif svd=="partial":    
            U, s, Vh = scislin.svds( X , k=k )   
            # reverse the n first columns of u
            U[ : , :k ] = U[ : , k-1::-1 ]
            # reverse s
            s = s[ ::-1 ]
            # reverse the n first rows of vt
            Vh[ :k , : ] = Vh[ k-1::-1 , : ]     
        
        elif svd=="trancated":
            U, s, Vh = sci.linalg.svd( X ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=False,
                                  check_finite=True)
            U = U[ : , range(k) ]
            s = s[ range(k) ]
            Vh = Vh[ range(k) , : ]
    
        else: 
            raise ValueError('SVD algorithm is not supported.')
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
    G = np.dot( Y , Vh.conj().T * s**-1)
    M = np.dot( U.conj().T , G )
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    l, W = np.linalg.eig( M )    
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if modes=='standard': 
        F = np.dot( U , W )    
    elif modes=='exact': 
        F = np.dot( G , W )
    elif modes=='exact_scaled':     
        F = np.dot((1/l) * G , W )
    else: 
        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if return_amplitudes==True:   
        b , _ , _ , _ = sci.linalg.lstsq( F , A[ : , 0 ])

        if order==True: 
            sort_idx = sorted(range(len(b.real)), key=lambda j: b[j], reverse=True) 
            F = F[  :, sort_idx ]
            b = b[ sort_idx ]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True: 
        V = np.fliplr(np.vander( l , N =  (n-1) ))     
        
        if order==True and return_amplitudes==True: 
            V = V[ sort_idx ,  : ]
    
    if return_amplitudes==True and return_vandermonde==True:
        return F, b, V
    elif return_amplitudes==True and return_vandermonde==False:
        return F, b,
    else:
        return F    
  
    #**************************************************************************   
    #End dmd
    #**************************************************************************  
     
     


def cdmd(A, k=None, c=None,  p=5, q=2, modes='standard', return_amplitudes=True,
        return_vandermonde=True, svd='rand', rsvd_type='fast', order=True):
    """
    Compressed Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `a` in space and time.
    The matrix `a` is decomposed as `a = FBV`, where the columns of `F`
    contain the dynamic modes. The modes are ordered corresponding
    to the amplitudes stored in the diagonal matrix `B`. `V` is a Vandermonde
    matrix describing the temporal evolution.

    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `a` with dimensions `(m, n)`.
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    c : float, [0,1]
        Parameter specifying the compression rate.         
    p : int, optional
        `p` sets the oversampling parameter for rSVD (default k=5).
    q : int, optional
        `q` sets the number of power iterations for rSVD (default=1).
    modes : str `{'standard', 'exact', 'exact_scaled'}`
        'standard' : uses the standard definition to compute the dynamic modes,
                    `F = U * W`.
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
        'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.
    svd : str `{'rand', 'partial', 'trancated'}`
        'rand' : uses randomized singular value decomposition (default). 
        'partial' : uses partial singular value decomposition.
        'trancated' : uses trancated singular value decomposition.
    rsvd_type : str `{'standard', 'fast'}`
        'standard' : (default) Standard algorithm as described in [1, 2]. 
        'fast' : Version II algorithm as described in [2].       
    order :  bool `{True, False}`
        True: return modes sorted according to the amplitudes

    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
    b_gpu : array_like
        1-D array containing the amplitudes of length `min(n-1, k)`.
    V : array_like
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.

    Notes
    -----
    

    References
    ----------
    S. L. Brunton, et al.
    "Compressed Sensing and Dynamic Mode Decomposition"
    arXiv preprint arXiv:1312.5186 (2013).
    
    J. H. Tu, et al.
    "On Dynamic Mode Decomposition: Theory and Applications."
    arXiv preprint arXiv:1312.0041 (2013).
    
    N. B. Erichson and C. Donovan.
    "Randomized Low-Rank Dynamic Mode Decomposition for Motion Detection"
    Under Review.    
    


    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************
 
    #Shape of A
    m, n = A.shape   

    #Compress
    if c==None:
        Ac = A
    else:
        r = np.ceil(m*(1-c))
        print "Compression rate %0.0f" %c
        print "Rows compressed from %0.0f" %m + " to %0.0f"  %r 
        S = np.array( np.random.uniform( 0 , 2 , size=( r , m ) ) )
        #S = np.array( np.random.standard_normal((n, m) ))

        Ac = np.dot( S , A )  

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = Ac[ : , range( 0 , n-1 ) ] #pointer
    Y = Ac[ : , range( 1 , n ) ] #pointer   
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    if k != None:
        if svd=="rand":
            U, s, Vh = rsvd( X, k=k , p=p , q=q , method=rsvd_type )  
        
        elif svd=="csvd":
            U, s, Vh = cSVD( X , k=k, q=q)  
        
        elif svd=="partial":    
            U, s, Vh = scislin.svds( X , k=k )   
            # reverse the n first columns of u
            U[ : , :k ] = U[ : , k-1::-1 ]
            # reverse s
            s = s[ ::-1 ]
            # reverse the n first rows of vt
            Vh[ :k , : ] = Vh[ k-1::-1 , : ]     
        
        elif svd=="trancated":
            U, s, Vh = sci.linalg.svd( X ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=False,
                                  check_finite=True)
            U = U[ : , range(k) ]
            s = s[ range(k) ]
            Vh = Vh[ range(k) , : ]
    
        else: 
            raise ValueError('SVD algorithm is not supported.')
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
    G = np.dot( Y , Vh.conj().T * s**-1)
    M = np.dot( U.conj().T , G )
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print M.shape
    l, W = np.linalg.eig( M )    
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
#    if modes=='standard': 
#        F = np.dot( U , W )    
#    elif modes=='exact': 
#        F = np.dot( G , W )
#    elif modes=='exact_scaled':     
#        F = np.dot((1/l) * G , W )
#    else: 
#        raise ValueError('Type of modes is not supported, choose "exact" or "standard".')

    F = np.dot( A[ : , range( 1 , n ) ] , np.dot(Vh.conj().T * s**-1, W ) )   
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute amplitueds b using least-squares: Fb=x1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if return_amplitudes==True:   
        b , _ , _ , _ = sci.linalg.lstsq( F , A[ : , 0 ])

        if order==True: 
            sort_idx = sorted(range(len(b.real)), key=lambda j: b[j], reverse=True) 
            F = F[  :, sort_idx ]
            b = b[ sort_idx ]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix (CPU)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True: 
        V = np.fliplr(np.vander( l , N =  (n-1) ))     
        
        if order==True and return_amplitudes==True: 
            V = V[ sort_idx ,  : ]
    
    if return_amplitudes==True and return_vandermonde==True:
        return F, b, V
    elif return_amplitudes==True and return_vandermonde==False:
        return F, b,
    else:
        return F    
  
    #**************************************************************************   
    #End cDMD
    #**************************************************************************  


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
        'standard' : Standard algorithm as described in [1, 2]
        'fast' : Version II algorithm as described in [2]                 
    sdist : str `{'unif', 'norm'}`
    
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
    >>> import numpy as np
    >>> import scipy as sci

    
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
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Generate a random sampling matrix O
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sdist=='unif':   
        O = np.array( np.random.uniform( 0 , 1 , size=(n, k+p) ) , dtype = dat_type ) 
        if isreal==False: 
            O += 1j * np.array( np.random.uniform(-1 , 1 , size=(n, k+p) ) , dtype=real_type )
     
    if sdist=='norm':   
        O = np.array( np.random.standard_normal( size=(n, k+p) ) , dtype = dat_type ) 
        if isreal==False: 
            O += 1j * np.array( np.random.standard_normal( size=(n, k+p) ) , dtype=real_type )     
     
    if sdist=='sparse':   
        sf = np.sqrt(n)       
        #sf = n / np.log(n)
        O = np.array( np.random.choice( a=np.array( [1,0,1] ) , p=np.array( [ 1./(2.*sf), 1. - 1./sf , 1./(2.*sf)] ), size=(n, k+p), replace=True ) , dtype = dat_type) 

    if sdist=='sobol':   
        O = np.array( i4_sobol_generate(k+p, N) , dtype = dat_type) 
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Build sample matrix Y : Y = A * O
    #Note: Y should approximate the range of A
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    Y = np.dot( A , O )

    if sdist=='sparse': 
        Y = Y * np.sqrt(sf)
    
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
    print 'Import rla'