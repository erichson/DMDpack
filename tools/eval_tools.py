import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sklearn as sklearn



#Tools
#-----------------------------------------------------------------------------

def evalplot(I, BG, GT, Dim, N, name, color, linestyle):
    #~~~~~~~~~~
    #Compute ROC curve, area under the curve and F-measure
    #~~~~~~~~~~~~~~~~~~~~~   
    Diff = np.int32(np.rint(np.abs(I-BG)))
    #Diff = np.abs(I-BG)
    fpr, tpr, thresholds = roc_curve(GT.reshape(Dim*N), Diff.reshape(Dim*N), pos_label=1)
    AUC = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(GT.reshape(Dim*N), Diff.reshape(Dim*N), pos_label=1)
    F = 2*(recall*precision)/(recall+precision)
    where_are_NaNs = np.isnan(F)
    F[where_are_NaNs] = 0
    Fmax = F.max()
    print "Area under the ROC curve " + name + " : %0.3f" % AUC
    print "F-measure " + name + " : %0.3f" % Fmax
    print "Best threshold: %0.3f " % (thresholds[np.argmax(F)])
    plt.subplot(121)
    plt.plot(fpr, tpr, color=color, linestyle=linestyle, label= name + ' : AUC=%0.3f' % AUC ,  linewidth=3)
    plt.subplot(122)
    plt.plot(thresholds, F[0:len(thresholds)] ,color=color, linestyle=linestyle, label= name + ' F=%0.3f' % Fmax,  linewidth=3)
    del(fpr, tpr, thresholds, precision, recall, F)
    
    # Plop settings
    plt.subplot(121)
    plt.xlim([0, 0.3])
    plt.ylim([0.6, 0.9])
    plt.xlabel('1-specificity', fontsize=25)
    plt.ylabel('recall', fontsize=25)
    #plt.title('ROC')
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True)    
    
    plt.subplot(122)
    plt.xlim([20, 70])
    plt.ylim([0.6, 0.8])
    plt.xlabel('Threshold', fontsize=25)
    plt.ylabel('F-Measure', fontsize=25)
    #plt.title('ROC')
    plt.legend(loc="upper right", fontsize=20)
    plt.grid(True)




def bgMETRIC(GT, CB):
    m, n = GT.shape
    Diff = np.abs(GT-CB)   
    #AGE  
    AGE = np.mean(np.mean(Diff))    
    print "AGE: %0.2f" % AGE  
    #EPs and pEPs
    threshold=20
    Errors=Diff>threshold
    EPs=np.sum(np.sum(Errors))
    pEPs=EPs/(m*n)
    print "EPs: %0.2f" % EPs 
    print "pEPs: %0.2f" % (pEPs*100) 
    
    

def plot_modes( V ):
    m, n = V.shape
    omega = np.log(V1[:,1])
    
    labels = ['mode{0}'.format(i) for i in range(m)]
    plt.subplots_adjust(bottom = 0.1)
    #vert line
    plt.axvline(x=0,color='k',ls='dashed', lw=2)
    #horiz line
    plt.axhline(y=0,color='k',ls='dashed', lw=2)

    #plot omega    
    plt.scatter( omega.real, omega.imag, marker = 'o', c = 'r', s=20*4 )

    #plot labels  
    for label, x, y in zip(labels, omega.real, omega.imag):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.3),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0'))
    
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel('omega real', fontsize=20)
    plt.ylabel('omega imag', fontsize=20)
    
    plt.show()