from __future__ import division
import numpy as np
import scipy as sci
import scipy.sparse.linalg as scislin
from numpy.testing import assert_raises

from skrla import *


def test():
    
      #Test 1: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=0, q=0, method='standard')
      assert np.allclose(a, np.dot(U, np.dot(np.diag(s), Vh)), 1e-4)

      #Test 2: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=0, q=0, method='fast')
      assert np.allclose(a, np.dot(U, np.dot(np.diag(s), Vh)), 1e-4)
      
      #Test 3: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=0, q=2, method='standard')
      assert np.allclose(a, np.dot(U * s, Vh), 1e-4)
      
      #Test 4: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=0, q=2, method='fast')
      assert np.allclose(a, np.dot(U * s, Vh), 1e-4) 
      
      #Test 5: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=5, q=0, method='standard')
      assert np.allclose(a, np.dot(U * s, Vh), 1e-4)
      
      #Test 6: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=5, q=0, method='fast')
      assert np.allclose(a, np.dot(U * s, Vh), 1e-4) 
      
      #Test 5: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float32) + 1j*np.array(np.random.rand(n,m), np.float32)
      U, s, Vh = rsvd(a, k=n, p=0, q=0, method='standard')
      assert np.allclose(a, np.dot(U * s, Vh), 1e-4)
      
      #Test 6: rsvd
      m,n = 9,8
      a=np.array(np.random.rand(n,m), np.float64) + 1j*np.array(np.random.rand(n,m), np.float64)
      U, s, Vh = rsvd(a, k=n, p=0, q=0, method='fast')
      assert np.allclose(a, np.dot(U * s, Vh), 1e-4) 
      
      
      
      #Test 1: dmd
      m, n = 9, 7
      a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)), 
                     np.complex128, order='F')
      f, b, v = dmd(a, k=None, p=0, q=1, modes='standard', svd='rand')
      assert np.allclose(a[:,:(n-1)], np.dot(f, np.dot(np.diag(b), v) ), 1e-4)
      
      #Test 1: dmd
      m, n = 9, 7
      a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)), 
                     np.complex128, order='F')
      f, b, v = dmd(a, k=(n-1), p=0, q=1, modes='standard', svd='rand')
      assert np.allclose(a[:,:(n-1)], np.dot(f, np.dot(np.diag(b), v) ), 1e-4)
      
      
      #Test 1: dmd
      m, n = 9, 7
      a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)) + 1j*np.fliplr(np.vander(np.random.rand(m)+1, n)), 
                     np.complex128, order='F')
      f, b, v = dmd(a, k=(n-1), p=0, q=1, modes='standard', svd='trancated')
      assert np.allclose(a[:,:(n-1)], np.dot(f, np.dot(np.diag(b), v) ), 1e-4)      
      
      #Test 1: dmd
      m, n = 9, 7
      a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float64, order='C')   
      f, b, v = dmd(a, k=(n-2), p=0, q=1, modes='standard', svd='rand')
      assert np.allclose(a[:,:(n-1)], np.dot(f, np.dot(np.diag(b), v) ), 1e-4)
      
      #Test 1: dmd
      m, n = 9, 7
      a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float64, order='C') 
      f, b, v = dmd(a, k=(n-2), p=0, q=1, modes='standard', svd='trancated')
      assert np.allclose(a[:,:(n-1)], np.dot(f, np.dot(np.diag(b), v) ), 1e-4)      
      
      #Test 1: dmd
      m, n = 9, 7
      a = np.array(np.fliplr(np.vander(np.random.rand(m)+1, n)), np.float64, order='C') 
      f, b, v = dmd(a, k=(n-2), p=0, q=1, modes='standard', svd='partial', rsvd_type='standard')
      assert np.allclose(a[:,:(n-1)], np.dot(f, np.dot(np.diag(b), v) ), 1e-4) 

if __name__ == "__main__":
    test()