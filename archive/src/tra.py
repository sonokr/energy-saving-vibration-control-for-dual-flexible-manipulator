def tra(a,S):
    c=np.array([1.0,-1.0])
    sig=10.**(-a[2:4])
    t=np.linspace(0., TE, 2*Nte+1)
    T=-1+2*t/TE
    abe1=np.abs((1 - T**2)/np.exp((-c[0] + T)**2/sig[0]))
    abe2=np.abs((1 - T**2)/np.exp((-c[1] + T)**2/sig[1]))
    W=np.array([a[0]/abe1.max(), a[1]/abe2.max()])
    #
    #S=np.zeros([2*Nrk+1,3])
    for i in range(0,2*Nte+1):
        net=t[i]/TE
        dnet=1./TE
        ddnet=0.0
        for j in range(0,2):
            net=net+(W[j]*(1 - T[i]**2))/np.exp((-c[j] + T[i])**2/sig[j])
            dnet=dnet+(-4.*W[j]*(-c[j] + T[i]*(1 + sig[j] + (c[j] -\
             T[i])*T[i])))/(np.exp((c[j] - T[i])**2/sig[j])*sig[j]*TE)
            ddnet=ddnet+(-8.*W[j]*(-2.*c[j]**2+sig[j]+sig[j]**2+T[i]*\
             (4.*c[j]*(1+sig[j])+T[i]*(-2+2*c[j]**2-5*sig[j]+2*T[i]*\
             (-2*c[j]+T[i])))))/(np.exp((c[j]-T[i])**2/sig[j])*\
             sig[j]**2*TE**2)
        S[i,0]=SE*(net-np.sin(2*np.pi*net)/2/np.pi)
        S[i,1]=2*SE*np.sin(np.pi*net)**2*dnet
        S[i,2]=2*SE*np.sin(np.pi*net)*(2*np.pi*np.cos(np.pi*net)\
         *dnet**2+np.sin(np.pi*net)*ddnet)
    S[2*Nte+1:,0]=SE
    return S
