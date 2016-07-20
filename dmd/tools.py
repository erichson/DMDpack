"""

Dynamic Mode Decomposition (DMD) tools.

"""

from __future__ import division
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from numpy.testing import assert_raises
 
def plot_modes( omega, color='r', color2='blue', name=None, maker='o', alpha = 0.3, labelon=True, xytx=-20, xyty=20):
    
    m = len(omega)
    
    labels = ['mode{0}'.format(i) for i in range(m)]
    plt.subplots_adjust(bottom = 0.1)
    #vert line
    plt.axvline(x=0,color='k',ls='dashed', lw=2)
    #horiz line
    plt.axhline(y=0,color='k',ls='dashed', lw=2)

    #plot omega    
    plt.scatter( omega.real, omega.imag, marker = maker, c = color, s=20*9, label = name )

    #plot labels
    if labelon==True: 
        for label, x, y in zip(labels, omega.real, omega.imag):
            xytx2, xyty2 =  xytx,   xyty
            color2=np.array([0.4,  0.4,  1.])
            plt.annotate(
                    label, 
                    xy = (x, y), xytext = (xytx2, xyty2),
                    textcoords = 'offset points', ha = 'right', va = 'bottom', fontsize=12, color='white',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = color2, alpha = alpha),
                    arrowprops = dict(facecolor='black', shrink=0.11))
            
    
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel('Real', fontsize=25)
    plt.ylabel('Imaginary', fontsize=25)
    plt.tick_params(axis='y', labelsize=18) 
    plt.tick_params(axis='x', labelsize=18) 
    #if name != None: plt.legend(loc="lower right", fontsize=25)

    plt.show()
    


     
if __name__ == "__main__":
	print "Imported."

