# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#PARAMETERS

M=5000 #mémoire du système
time = 2000000

gamma_init=0.23 #facteur d'implicature

#np.random.seed(77)
#np.random.seed(43)
np.random.seed(23)

beta_min=0.863
beta_max=0.863

delta_min=0.0001
delta_max=0.0001


loop=250 #number of processes

size=1. #As the parameter size increases, the size of the relevant time window on which produced occurrences are counted diminishes.
M_W=int(M/size)
W=8 #Number of time windows on which the sliding average is computed.

##OPTIONS

error_threshold=0.1 #float number between 0. and 1., with 0. for no error tolerance, and 1. to accept all logit modelings. 

close=950. #Float number from 0 to 1000. This parameter governs the closeness to the channel of the minimal value taken as an input for the logit transformation. 0 corresponds to the x_bar value, 100 to the upper fluctuation.

plot_logit=0 #=1 to plot the logit transformation of each process.

alloc=0 #=1 to consider an alternative mechanism

both_mechs=0 #=1 # to consider both speaker/producer and hearer/interpreter mechanism

#INITIALIZING LISTS AND INDICES

j=0
j_ctrl=0

phiII=[]
pente=[]
logphi=[]
logP=[]
beta_data=[]
gamma_data=[]
delta_data=[]
x_bar_data=[]
t_1_data=[]
t_2_data=[]
t_init=[]
err_data=[]

##DEFINITIONS DE FONCTIONS##

def fun(x,a,b,c,x_bar):
    res=np.zeros_like(x)
    d=(a-c)*x_bar+b
    res[x<x_bar]=a*x[x<x_bar]+b
    res[x>=x_bar]=c*x[x>=x_bar]+d
    return res


def fun2(x,a,b,c,d,x_bar):
    if x<x_bar :
        return a*x+b
    else :
        return c*x+d

def alpha(x):
    return (np.arctanh(2.*x-1))**2

def linear(x,alp,bet):
    return alp*x+bet




##DEBUT DE LA BOUCLE##

init=0

while j<loop:

    beta=np.random.uniform(beta_min,beta_max)

    delta_gamma=np.random.uniform(max(0,delta_min),min(1,delta_max)) ##Ecart au gamma critique


    ##DETERMINATION DU GAMMA CRITIQUE

    g=gamma_init

    def balance(x,gam):
        return 1./2.*(1+np.tanh(beta*((x+gam)-(1-x))/np.sqrt((x+gam)*(1-x))))-x

    def balance_der(x,gam):
        return 1./4.*beta*(1+gam)**2*np.exp(-3./2.*np.log((x+gam)*(1-x)))*(1-(np.tanh(beta*((x+gam)-(1-x))/np.sqrt((x+gam)*(1-x))))**2)-1

    coeff_x = [1, 3*g-1, beta**2*(1+g)**4+3*g*(g-1), (g-3)*g**2, -g**3]

    r=np.roots(coeff_x)

    for i in range(len(r)):
        if r[i]==np.conjugate(r[i]):
            r[i]=r[i].real
            if (r[i]>0)&(r[i]<1):
                a=r[i]
                a=a.real
                break
    ctrl=0

    db=np.abs(balance(a,g))
    dder=np.abs(balance_der(a,g))
    
    while (db>10**(-6))|(dder>10**(-6)):

        if ctrl<10000: ##Une condition servant à éviter d'être coincé indéfiniment au cas où la récursion ne fonctionnerait pas. 
            
            coeff_g = [beta**2, 2*beta**2*(2*a-1)-(1-a)*alpha(a), beta**2*(2*a-1)**2-a*(1-a)*alpha(a)]
            
            r_g=np.roots(coeff_g)
            
            for i in range(len(r_g)):
                if r_g[i]==np.conjugate(r_g[i]):
                    r_g[i]=r_g[i].real
                    if (r_g[i]>0)&(r_g[i]<1):
                        g=r_g[i]
                        g=g.real
                        break

            coeff_x = [1, 3*g-1, beta**2*(1+g)**4+3*g*(g-1), (g-3)*g**2, -g**3]

            r=np.roots(coeff_x)

            for i in range(len(r)):
                if r[i]==np.conjugate(r[i]):
                    r[i]=r[i].real
                    if (r[i]>0)&(r[i]<1):
                        a=r[i]
                        a=a.real
                        break
            db=np.abs(balance(a,g))
            dder=np.abs(balance_der(a,g))

            g_look=g

            ctrl+=1

        else:
            break

    if both_mechs==1:
        g=np.sqrt(1+g)-1



    ##GAMMA CRITIQUE DETERMINE##
        
    gamma=g+delta_gamma
    

    F=[]
    P=[]
    xvec=[]
    for ix in range(1,1000):
        x=float(ix)/1000.
        xvec.append(x)
        if (alloc==0)|(both_mechs==1):
            f=(x+gamma)/(1.+gamma)
        else:
            f=x
        phi=(2*f-1)/np.sqrt(f*(1-f))
        if (alloc==0)&(both_mechs==0):
            F.append(1./2.*(1+np.tanh(beta*phi))-x)
        else:
            F.append(1./2.*(1./2.*(1+np.tanh(beta*phi))-x+gamma*(1-x)))
        P.append(1./2.*(1+np.tanh(beta*phi)))

        
    F=F[0:500]
    x_min=xvec[F.index(min(F))]
    P_min=P[F.index(min(F))]
    P_entry=P_min-np.sqrt(P_min*(1-P_min)/float(M_W))
    P_out=P_min+np.sqrt(P_min*(1-P_min)/float(M_W))
    ind_entry=min( range( len(P) ), key=lambda i:abs(P_entry-P[i]) )
    ind_out=min( range( len(P) ), key=lambda i:abs(P_out-P[i]) )

    x_out=P_out
    x_entry=P_entry


    memoire=[0]*M
    N=0
    x=float(N)/float(M)

    N_tot=[]

    N_M=0

    site_call=0

    count=0
    countdown=0


    for i in range(time):
      

        if (alloc==0)&(both_mechs==0):
            coin_site=1.
        else:
            coin_site=np.random.uniform(0,1)
            coin_add=np.random.uniform(0,1)

        if coin_site>0.5:
            site_call+=1
            if (alloc==1)&(both_mechs==0):
                f=x
            else:
                f=(x+gamma)/(1+gamma)
            if f!=0.0:
                if f!=1.:
                    phi=(2*f-1)/np.sqrt(f*(1-f))
                    P=1./2.*(1+np.tanh(beta*phi))
                else:
                    P=1.0
            else:
                P=0.0
            dice=np.random.uniform(0,1)
            if P>dice:
                N+=1
                memoire.append(1)
                N_M+=1
            else:
                memoire.append(0)
            erased=np.random.randint(0,M)
            occ=memoire[erased]
            if occ==1:
                N-=1
            memoire.remove(occ)
            x=float(N)/float(M)
        else :
            if coin_add<gamma:
                N+=1
                memoire.append(1)
                erased=np.random.randint(0,M)
                occ=memoire[erased]
                if occ==1:
                    N-=1
                memoire.remove(occ)
                x=float(N)/float(M)



        if i!=0:
            if i%M_W==0:
                x_occ=float(N_M)/float(site_call)
                N_tot.append(x_occ)
                N_M=0
                site_call=0
                if (x_occ>(1.-1./float(M))):
                    count=1
                if count==1:
                    countdown+=1
                if countdown>W:
                        break  




    N_W=[]
    
    for k in range(len(N_tot)-(W-1)):
        N_W.append((sum(N_tot[k:k+W]))/float(W))

    Nmax=max(N_W)

    if Nmax>(1.-10./float(M)):

            if (db<10**(-6))&(dder<10**(-6)):

                close_loc=1001.

                err=2.

                while (close_loc>close)&(err>error_threshold):

                    close_loc-=1.

                    logit2=[]
                    bit=1
                    inibit=1
                    lower_x=x_out
                    lower_x2=(1.*(1000.-close_loc)+close_loc*lower_x)/1000.
                    for i in range(len(N_W)):
                        if inibit==1:
                            if N_W[i]>x_entry:
                                t_init.append(i)
                                inibit=0
                        if N_W[i]<(1.-10./float(M)):
                            if N_W[i]>lower_x2:
                                logit2.append(1./2.*np.log((N_W[i]-lower_x2)/(1.-N_W[i])))
                                if bit==1:
                                    t_1_data.append(i)
                                    bit=0
                            if N_W[i]<lower_x2:
                                if not(not(logit2)):
                                    t_1_data.pop(-1)
                                    bit=1
                                logit2=[]
                        else:
                            t_2_data.append(i-t_1_data[-1])
                            t_1_data[-1]=t_1_data[-1]-t_init[-1]
                            break

                    #logit2=np.array(logit2[:-2])

                    logit2=np.array(logit2)

                    w_MB=logit2.shape[0]

                    abs_2=np.arange(w_MB)

                    para_logit2,cov_logit2=opt.curve_fit(linear,abs_2,logit2,[1.,0.])

                    h_MB, ord_MB = para_logit2

                    err=sum((logit2-(h_MB*abs_2+ord_MB))**2)/sum((logit2-(sum(logit2)/w_MB))**2)

                if plot_logit==1:

                    length=len(N_W)
                    low1=[]
                    low2=[]

                    for i in range(length):
                        low1.append(lower_x)
                        low2.append(lower_x2)

                    xdumb=np.arange(length)
                    
                    plt.plot(N_W)
                    plt.plot(xdumb,low1)
                    plt.plot(xdumb,low2)
                    plt.show()

                    yP3=linear(abs_2,h_MB,ord_MB)

                    plt.plot(abs_2, logit2, 'go', lw=2)
                    plt.plot(abs_2,yP3,'r',lw=2)
                    plt.show()

                if err<error_threshold:

                    phiII.append(w_MB)
                    pente.append(h_MB)
                    logphi.append(np.log(w_MB))
                    logP.append(np.log(h_MB))

                    beta_data.append(beta)
                    gamma_data.append(gamma)
                    delta_data.append(delta_gamma)

                    err_data.append(err)

                    j+=1

                else :
                    print err
        

    if j%10==0:
        print j

    j_ctrl+=1


##Fit symétrique

mcov=np.cov(logP,logphi)
trA=np.trace(mcov)
detA=np.linalg.det(mcov)
lambdaA=trA/2.*(1-np.sqrt(1-4.*detA/(trA**2)))
ob=mcov[0,1]
oa=mcov[0,0]
oc=mcov[1,1]

coeff_P=np.sqrt(ob**2/(ob**2+(lambdaA-oa)**2))
coeff_phi=(lambdaA-oa)/ob*coeff_P
coeff_cross=-coeff_P*np.mean(logP)-coeff_phi*np.mean(logphi)

nu=-coeff_phi/coeff_P




