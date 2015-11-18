"""

Dynamic Mode Decomposition (DMD) python function.

"""

from __future__ import division
import numpy as np
import scipy as sci
import scipy.sparse.linalg as scislin
from numpy.testing import assert_raises
 
from rla import rsvd
    

def dmd(A, dt = 1, k=None, p=5, q=2, modes='exact',
        return_amplitudes=False, return_vandermonde=False, 
        svd='partial', rsvd_type='fast', sdist='punif', order=True):
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
    
    dt : scalar or array_like  
        Factor specifying the time difference between the observations.      
    
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    
    p : int, optional
        `p` sets the oversampling parameter for rSVD (default `p=5`).
    
    q : int, optional
        `q` sets the number of power iterations for rSVD (default `q=1`).
    
    modes : str `{'standard', 'exact', 'exact_scaled'}`
        'standard' : uses the standard definition to compute the dynamic modes, `F = U * W`.
        
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
        
        'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.
    
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.
    
    svd : str `{'rsvd', 'partial', 'truncated'}`
        'rsvd' : uses randomized singular value decomposition (default). 
        
        'partial' : uses partial singular value decomposition.
        
        'truncated' : uses trancated singular value decomposition.
    
    rsvd_type : str `{'standard', 'fast'}`
        'standard' : (default) Standard algorithm as described in [1, 2]. 
        
        'fast' : Version II algorithm as described in [2].  
    
    sdist : str `{'unif', 'punif', 'norm', 'sparse'}`
        'unif' : Uniform `[-1,1]`.
        
        'punif' : Uniform `[0,1]`.
        
        'norm' : Normal `~N(0,1)`.
        
        'sparse' : Sparse uniform `[-1,1]`.
    
    order :  bool `{True, False}`
        True: return modes sorted.


    Returns
    -------
    F : array_like
        Matrix containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.
   
    b : array_like, optional
        1-D array containing the amplitudes of length `min(n-1, k)`.
    
    V : array_like, optional
        Vandermonde matrix of shape `(n-1, n-1)`  or `(k, n-1)`.
    
    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`. 


    Notes
    -----


    References
    ----------
    J. H. Tu, et al.
    "On Dynamic Mode Decomposition: Theory and Applications" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).   
    
    N. B. Erichson and C. Donovan.
    "Randomized Low-Rank Dynamic Mode Decomposition for Motion Detection" (2015).
    Under Review.    
    
    
    Examples
    --------
    >>> #Numpy
    >>> import numpy as np
    >>> #DMD
    >>> from skrla import dmd
    >>> #Plot libs
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> from matplotlib import cm
    
    >>> #
    >>> # Create an artifical data-set:
    >>> #
    >>> # Define time and space discretizations
    >>> x=np.linspace( -15, 15, 200)
    >>> t=np.linspace(0, 8*np.pi , 80) 
    >>> dt=t[2]-t[1]
    >>> X, T = np.meshgrid(x,t)
    >>> # Create two patio-temporal patterns
    >>> F1 = 0.5* np.cos(X)*(1.+0.* T)
    >>> F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
    >>> # Add both signals
    >>> F = (F1+F2)
    
    >>> #Plot dataset
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(231, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F')
    >>> ax = fig.add_subplot(232, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F1')
    >>> ax = fig.add_subplot(233, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X, T, F2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F2')
    
    >>> #Dynamic Mode Decomposition of F
    >>> F_gpu = np.array(F.T, np.complex64, order='F')
    >>> F_gpu = gpuarray.to_gpu(F_gpu) 
    >>> Fmodes, b, V, omega = dmd(F, k=2, modes='exact', return_amplitudes=True, return_vandermonde=True)
    >>> omega = omega_gpu.get()

    >>> #Reconstruct the original signal
    >>> plt.scatter(omega.real, omega.imag, marker='o', c='r')
    >>> F1tilde = np.dot(Fmodes[:,0:1] , np.dot(b[0], V[0:1,:] ) )
    >>> F2tilde = np.dot(Fmodes[:,1:2] , np.dot(b[1], V[1:2,:] ) )
    
    >>> #Plot DMD modes
    >>> #Mode 0
    >>> ax = fig.add_subplot(235, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X[0:F1tilde.shape[1],:], T[0:F1tilde.shape[1],:], F1tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F1_tilde')
    >>> #Mode 1
    >>> ax = fig.add_subplot(236, projection='3d')
    >>> ax = fig.gca(projection='3d')
    >>> surf = ax.plot_surface(X[0:F2tilde.shape[1],:], T[0:F2tilde.shape[1],:], F2tilde.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    >>> ax.set_zlim(-1, 1)
    >>> plt.title('F2_tilde')
    >>> plt.show()     


    """

    #*************************************************************************
    #***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
    #***                              <2015>                               ***
    #***                       License: BSD 3 clause                       ***
    #*************************************************************************
 
    #Shape of D
    m, n = A.shape   
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
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = A[ : , range( 0 , n-1 ) ] #pointer
    Y = A[ : , range( 1 , n ) ] #pointer   
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    if k != None:
        if svd=="rsvd":
            U, s, Vh = rsvd( X, k=k , p=p , q=q , sdist=sdist, method=rsvd_type )  
        
        elif svd=="partial":    
            U, s, Vh = scislin.svds( X , k=k )   
            # reverse the n first columns of u
            U[ : , :k ] = U[ : , k-1::-1 ]
            # reverse s
            s = s[ ::-1 ]
            # reverse the n first rows of vt
            Vh[ :k , : ] = Vh[ k-1::-1 , : ]     
        
        elif svd=="truncated":
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

    Vscaled = Vh.conj().T * s**-1
    G = np.dot( Y , Vscaled ) 
    M = np.dot( U.conj().T , G )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    l, W = sci.linalg.eig( M , right=True, overwrite_a=True )    

    omega = np.log(l) / dt
 
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


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Compute Vandermonde matrix
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if return_vandermonde==True: 
        V = np.fliplr(np.vander( l , N =  (n-1) ))     
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Order
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if order==True: 
        sort_idx = sorted(range(len(omega.real)), key=lambda j: np.abs(omega[j]), reverse=False) 
        F = F[  :, sort_idx ]
        omega = omega[ sort_idx ]  
        if return_amplitudes==True: b = b[ sort_idx ]
        if return_vandermonde==True: V = V[ sort_idx ,  : ]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Return 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if return_amplitudes==True and return_vandermonde==True:
        return F, b, V, omega
    elif return_amplitudes==True and return_vandermonde==False:
        return F, b, omega
    else:
        return F, omega    
  
    #**************************************************************************   
    #End dmd
    #**************************************************************************  
     
     


def cdmd(A, dt = 1, k=None, c=None, sdist='norm', sf=3, p=5, q=2, modes='exact',
         return_amplitudes=False, return_vandermonde=False, svd='rsvd', 
         rsvd_type='fast', order=True, trace=True):
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
    
    dt : scalar or array_like  
        Factor specifying the time difference between the observations.  
    
    k : int, optional
        If `k < (n-1)` low-rank Dynamic Mode Decomposition is computed.
    
    c : float, [0,1]
        Parameter specifying the compression rate.         
    
    sdist : str `{'unif', 'punif', 'norm', 'sparse'}`  
        Specify the distribution of the sensing matrix `S`. 
    
    sf : int, optional
        `sf` sets the sparsity factor for the `sparse` sdist, i.e. `sf=0.9` means
        `90%` of the sensing matrix `S` entries are zero.
    
    p : int, optional
        `p` sets the oversampling parameter for rSVD (default `k=5`).
    
    q : int, optional
        `q` sets the number of power iterations for rSVD (`default=1`).
    
    modes : str `{'exact', 'exact_scaled'}`
        'exact' : computes the exact dynamic modes, `F = Y * V * (S**-1) * W`.    
        
        'exact_scaled' : computes the exact dynamic modes, `F = (1/l) * Y * V * (S**-1) * W`.
    
    return_amplitudes : bool `{True, False}` 
        True: return amplitudes in addition to dynamic modes. 
    
    return_vandermonde : bool `{True, False}`
        True: return Vandermonde matrix in addition to dynamic modes and amplitudes.
    
    svd : str `{'rsvd', 'partial', 'truncated'}`
        'rand' : uses randomized singular value decomposition (default). 
        
        'partial' : uses partial singular value decomposition.
        
        'truncated' : uses trancated singular value decomposition.
    
    rsvd_type : str `{'standard', 'fast'}`
        'standard' : (default) Standard algorithm as described in [1, 2]. 
        
        'fast' : Version II algorithm as described in [2].       
    
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
    #Compress
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if c==None:
        Ac = A
    else:
        if trace==True: print "Rows compressed from %0.0f" %m + " to %0.0f"  %c

        if sdist=='unif':   
            S = np.array( np.random.uniform( -1 , 1 , size=( c, m ) ) , dtype = dat_type ) 
            if isreal==False: 
                S += 1j * np.array( np.random.uniform(-1 , 1 , size=( c, m ) ) , dtype = dat_type )
        
        if sdist=='punif':   
            S = np.array( np.random.uniform( 0 , 1 , size=( c, m ) ) , dtype = dat_type ) 
            if isreal==False: 
                S += 1j * np.array( np.random.uniform(0 , 1 , size=( c, m ) ) , dtype = dat_type )
            
        elif sdist=='norm':   
            S = np.array( np.random.standard_normal( size=( c, m ) ) , dtype = dat_type ) 
            if isreal==False: 
                S += 1j * np.array( np.random.standard_normal( size=( c, m ) ) , dtype = dat_type )     
            
        elif sdist=='sparse':   
            density = 1-sf
            S = sci.sparse.rand(c, m, density=density, format='coo', dtype=dat_type, random_state=None)
            if isreal==False: 
                S.data += 1j * np.array( np.random.uniform(0 , 1 , size=( len(S.data) ) ) , dtype = dat_type )
            if trace==True: print "Sparse: %f" %sf + " zeros "
            S.data *=2          
            S.data -=1            
            S = S.tocsr()
            
        #Compress input matrix
        Ac = S.dot(A)    
        del(S)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Split data into lef and right snapshot sequence
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    X = Ac[ : , range( 0 , n-1 ) ] #pointer
    Y = Ac[ : , range( 1 , n ) ] #pointer   
     
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Singular Value Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    if k != None:
        if svd=="rsvd":
            U, s, Vh = rsvd( X, k=k , p=p , q=q , method=rsvd_type )  
        
        elif svd=="partial":    
            U, s, Vh = scislin.svds( X , k=k )   
            # reverse the n first columns of u
            U[ : , :k ] = U[ : , k-1::-1 ]
            # reverse s
            s = s[ ::-1 ]
            # reverse the n first rows of vt
            Vh[ :k , : ] = Vh[ k-1::-1 , : ]     
        
        elif svd=="truncated":
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
    Vscaled = Vh.conj().T * s**-1
    G = np.dot( Y , Vscaled )
    M = np.dot( U.conj().T , G ) 
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Eigen Decomposition
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    l, W = sci.linalg.eig( M , right=True, overwrite_a=True )    
 
    omega = np.log(l) / dt
         
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #Compute DMD Modes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if modes=='exact': 
        F = np.dot( A[ : , range( 1 , n ) ] , np.dot( Vscaled , W ) )   
    elif modes=='exact_scaled':  
        F = np.dot( A[ : , range( 1 , n ) ] , np.dot( Vscaled , W ) ) * ( 1/l )
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
        V = np.fliplr(np.vander( l , N =  (n-1) ))     
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Order
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if order==True: 
        sort_idx = sorted(range(len(omega.real)), key=lambda j: np.abs(omega[j]), reverse=False) 
        F = F[  :, sort_idx ]
        omega = omega[ sort_idx ]  
        if return_amplitudes==True: b = b[ sort_idx ]
        if return_vandermonde==True: V = V[ sort_idx ,  : ]

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

     
if __name__ == "__main__":
	print "Imported."

