from __future__ import division
import numpy as np
import scipy as sci
import scipy.sparse.linalg as scislin

from dmd import *

from unittest import main, makeSuite, TestCase, TestSuite
from numpy.testing import assert_raises

atol_float32 = 1e-4
atol_float64 = 1e-8



#
#******************************************************************************
#

class test_rlinalg(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_rsvd_float32(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float32, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='standard')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float32
       
    def test_rsvd_float64(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float64, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='standard')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float64     
        
    def test_rsvd_complex64(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float32, order='C') + 1j * np.array(np.random.randn(m, k), np.float32, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='standard')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float32    
        
    def test_rsvd_complex128(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float64, order='C') + 1j * np.array(np.random.randn(m, k), np.float64, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='standard')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float64

    def test_rsvdf_float32(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float32, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='fast')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float32
        
    def test_rsvdf_float64(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float64, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='fast')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float64    
        
    def test_rsvdf_complex64(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float32, order='C') + 1j * np.array(np.random.randn(m, k), np.float32, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=3, method='fast')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float32    
        
    def test_rsvdf_complex128(self):
        m, k = 100, 20
        A = np.array(np.random.randn(m, k), np.float64, order='C') + 1j * np.array(np.random.randn(m, k), np.float64, order='C')
        A = A.dot(A.T)
        U, s, Vt = rsvd(A, k=20, p=5, q=2, method='fast')
        Ak = U.dot(np.diag(s).dot(Vt))
        percent_error = 100*np.linalg.norm(A - Ak,'fro')/np.linalg.norm(Ak,'fro')
        assert percent_error < atol_float64
        

#
#******************************************************************************
#
        
        
class test_dmd(TestCase):
    def setUp(self):
        np.random.seed(123)        
        
    def test_dmd_standard(self):
        # Define time and space discretizations
        x=np.linspace( -10, 10, 100)
        t=np.linspace(0, 8*np.pi , 60) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        A = np.array((F1+F2).T, np.complex128, order='C')
        
        Fmodes, b, V, omega = dmd(A, k=2, modes='standard', svd='partial', return_amplitudes=True, return_vandermonde=True)
        Atilde = Fmodes.dot( np.dot(np.diag(b), V))        
        assert np.allclose(A, Atilde, atol_float64)   

    def test_dmd_exact(self):
        # Define time and space discretizations
        x=np.linspace( -10, 10, 100)
        t=np.linspace(0, 8*np.pi , 60) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        A = np.array((F1+F2).T, np.complex128, order='C')
        
        Fmodes, b, V, omega = dmd(A, k=2, modes='exact', svd='partial', return_amplitudes=True, return_vandermonde=True)
        Atilde = Fmodes.dot( np.dot(np.diag(b), V))        
        assert np.allclose(A, Atilde, atol_float64)  

    def test_rdmd_standard(self):
        # Define time and space discretizations
        x=np.linspace( -10, 10, 100)
        t=np.linspace(0, 8*np.pi , 60) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        A = np.array((F1+F2).T, np.complex128, order='C')
        
        Fmodes, b, V, omega = dmd(A, k=2, modes='standard', svd='rsvd', p=2, q=2, return_amplitudes=True, return_vandermonde=True)
        Atilde = Fmodes.dot( np.dot(np.diag(b), V))        
        assert np.allclose(A, Atilde, atol_float64) 

    def test_rdmd_exact(self):
        # Define time and space discretizations
        x=np.linspace( -10, 10, 100)
        t=np.linspace(0, 8*np.pi , 60) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        A = np.array((F1+F2).T, np.complex128, order='C')
        
        Fmodes, b, V, omega = dmd(A, k=2, modes='exact', svd='rsvd', p=2, q=2, return_amplitudes=True, return_vandermonde=True)
        Atilde = Fmodes.dot( np.dot(np.diag(b), V))        
        assert np.allclose(A, Atilde, atol_float64) 
     
    def test_cdmd_exact(self):
        # Define time and space discretizations
        x=np.linspace( -10, 10, 100)
        t=np.linspace(0, 8*np.pi , 60) 
        dt=t[2]-t[1]
        X, T = np.meshgrid(x,t)
        # Create two patio-temporal patterns
        F1 = 0.5* np.cos(X)*(1.+0.* T)
        F2 = ( (1./np.cosh(X)) * np.tanh(X)) *(2.*np.exp(1j*2.8*T))
        A = np.array((F1+F2).T, np.complex128, order='C')
        
        Fmodes, b, V, omega = cdmd(A, k=2, modes='exact', svd='truncated', c=10, return_amplitudes=True, return_vandermonde=True)
        Atilde = Fmodes.dot( np.dot(np.diag(b), V))        
        assert np.allclose(A, Atilde, atol_float64) 


#
#******************************************************************************
#
        
def suite():
    s = TestSuite()
    s.addTest(test_rlinalg('test_rsvd_float32'))
    s.addTest(test_rlinalg('test_rsvd_float64'))
    s.addTest(test_rlinalg('test_rsvd_complex64'))
    s.addTest(test_rlinalg('test_rsvd_complex128'))
    s.addTest(test_rlinalg('test_rsvdf_float32'))
    s.addTest(test_rlinalg('test_rsvdf_float64'))
    s.addTest(test_rlinalg('test_rsvdf_complex64'))
    s.addTest(test_rlinalg('test_rsvdf_complex128'))
    
    s.addTest(test_dmd('test_dmd_standard'))
    s.addTest(test_dmd('test_dmd_exact'))
    s.addTest(test_dmd('test_rdmd_standard'))
    s.addTest(test_dmd('test_rdmd_exact'))
    s.addTest(test_dmd('test_cdmd_exact'))

    
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
