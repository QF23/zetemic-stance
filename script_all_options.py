
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
#delta_min=0.000002
delta_min=0.0001
delta_max=0.0001
#delta_min=0.002
#delta_max=0.05

loop=250 #number of processes

size=1.0 #As the parameter size increases, the size of the relevant time window on which produced occurrences are counted diminishes.
M_W=int(M/size)
W=8 #Number of time windows on which the sliding average is computed.

##OPTIONS

error_threshold=0.1 #float number between 0. and 1., with 0. for no error tolerance, and 1. to accept all logit modelings. 

close=950. #Float number from 0 to 1000. This parameter governs the closeness to the channel of the minimal value taken as an input for the logit transformation. 0 corresponds to the x_bar value, 100 to the upper fluctuation.

plot_logit=1 #=1 to plot the logit transformation of each process.

para_fixed=0 #=1 to fix the value of the parameters for all loops

window=1 #=1 to consider time windows as with the other observable

observable_P=1 #1 to switch to observable P instead of x

x_bar_version=0 #1 to use a numerical separation of the two phases

old_version=0 #1 to run the older version of the program with a zero minimum for the logit. Works only if x_bar_version is 1. 

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

    #pow_delta=np.random.uniform(delta_max,delta_min)

    delta_gamma=np.random.uniform(max(0,delta_min),min(1,delta_max)) ##Ecart au gamma critique

    #delta_gamma=10**(-pow_delta)

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

    if init==0:
        g_fixed=g
        delta_fixed=delta_gamma
        beta_fixed=beta
        init=1
        
    if para_fixed==1:
        g=g_fixed
        delta_gamma=delta_fixed
        beta=beta_fixed
        
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

    h_analytic=max(F)

        
    F=F[0:500]
    x_min=xvec[F.index(min(F))]
    P_min=P[F.index(min(F))]
    P_entry=P_min-np.sqrt(P_min*(1-P_min)/float(M_W))
    P_out=P_min+np.sqrt(P_min*(1-P_min)/float(M_W))
    ind_entry=min( range( len(P) ), key=lambda i:abs(P_entry-P[i]) )
    ind_out=min( range( len(P) ), key=lambda i:abs(P_out-P[i]) )
    x_entry=xvec[ind_entry]
    x_out=xvec[ind_out]

    if observable_P==1:
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

        if window==0:
            N_tot.append(x)
            if x==1.:
                break

        else:

            if i!=0:
                if i%M_W==0:
                    if observable_P==1:
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
                        
                    else:
                        N_tot.append(x)
        
            
    if window==1:

        N_W=[]
        
        for k in range(len(N_tot)-(W-1)):
            N_W.append((sum(N_tot[k:k+W]))/float(W))


    else:
        N_W = N_tot

    der_N=[N_W[i]-N_W[i-1] for i in range(1,len(N_W))]
    x_der=der_N.index(max(der_N))+1
    val_ref=N_W[x_der]


    Nmax=max(N_W)

    if x_bar_version==1:
        
        logit=[]
        for i in range(len(N_W)):
            if N_W[i]<(1.-1./float(M)):
                if N_W[i]>0.:
                    logit.append(1./2.*np.log(N_W[i]/(1.-N_W[i])))
            else:
                break

        x=np.arange(0,len(logit),1)*1.0
        logit=np.array(logit)


        parametres,covariance=opt.curve_fit(fun,x,logit,[0.00001,0,7.8,len(logit)*2./3.])

        a,b,c,x_bar=parametres[:]

        if old_version==1:    

            phaseII=len(logit)-x_bar

            if plot_logit==1:
                
                lower_x=x_out
                lower_x2=(1.*(1000.-close)+close*lower_x)/1000.
                lower_x_0=N_W[int(x_bar)-1]
                y2=fun(x,a,b,c,x_bar)
                plt.plot(x,y2)
                plt.plot(x,logit)
                plt.show()

                length=len(N_W)
                low1=[]
                low2=[]
                low3=[]

                for i in range(length):
                    low1.append(lower_x)
                    low2.append(lower_x2)
                    low3.append(lower_x_0)

                xdumb=np.arange(length)
                
                plt.plot(N_W)
                plt.plot(xdumb,low1)
                plt.plot(xdumb,low2)
                plt.plot(xdumb,low3)
                plt.show()    

            if Nmax>0.9 :

                if (db<10**(-6))&(dder<10**(-6)):

                    if x_bar>1.:

                        if a<c:
                        
                            if np.isfinite(np.log(phaseII)):

                                    phiII.append(phaseII)
                                    pente.append(c)
                                    logphi.append(np.log(phaseII))
                                    logP.append(np.log(c))
                                    beta_data.append(beta)
                                    gamma_data.append(gamma)
                                    delta_data.append(delta_gamma)
                                    x_bar_data.append(x_bar)

                                    j+=1

        else:

            if Nmax>(1.-10./float(M)):

                if (db<10**(-6))&(dder<10**(-6)):

                    logit2=[]
                    bit=1
                    inibit=1
                    lower_x=x_out
                    lower_x2=(1.*(1000.-close)+close*lower_x)/1000.
                    lower_x_bar=N_W[int(x_bar)-1]
                    for i in range(len(N_W)):
                        if inibit==1:
                            if N_W[i]>x_entry:
                                t_init.append(i)
                                inibit=0
                        if N_W[i]<(1.-10./float(M)):
                            if N_W[i]>lower_x_bar:
                                logit2.append(1./2.*np.log((N_W[i]-lower_x_bar)/(1.-N_W[i])))
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

                    logit2=np.array(logit2[:-2])

                    w_MB=logit2.shape[0]

                    abs_2=np.arange(w_MB)

                    para_logit2,cov_logit2=opt.curve_fit(linear,abs_2,logit2,[1.,0.])

                    h_MB, ord_MB = para_logit2

                    if plot_logit==1:

                        length=len(N_W)
                        low1=[]
                        low2=[]
                        low3=[]

                        for i in range(length):
                            low1.append(lower_x)
                            low2.append(lower_x2)
                            low3.append(lower_x_bar)

                        xdumb=np.arange(length)
                        
                        plt.plot(N_W)
                        plt.plot(xdumb,low1)
                        plt.plot(xdumb,low2)
                        plt.plot(xdumb,low3)
                        plt.show()

                        yP3=linear(abs_2,h_MB,ord_MB)

                        plt.plot(abs_2, logit2, 'go', lw=2)
                        plt.plot(abs_2,yP3,'r',lw=2)
                        plt.show()


                    phiII.append(w_MB+3.)
                    pente.append(h_MB)
                    logphi.append(np.log(w_MB))
                    logP.append(np.log(h_MB))

                    beta_data.append(beta)
                    gamma_data.append(gamma)
                    delta_data.append(delta_gamma)

                    j+=1
                    
    else:
        
        if Nmax>(1.-10./float(M)):

                if (db<10**(-6))&(dder<10**(-6)):

                    close_loc=1001.

                    err=2.

                    while (close_loc>close)&(err>error_threshold):

                        close_loc-=1.

                        logit2=[]
                        bit=1
                        inibit=1
                        upper_x=1.-10./float(M)
                        lower_x=(1.*(1000.-close_loc)+close_loc*x_out)/1000.
                        for i in range(len(N_W)):
                            if inibit==1:
                                if N_W[i]>x_entry:
                                    t_init.append(i)
                                    inibit=0
                            if N_W[i]<upper_x:
                                if N_W[i]>lower_x:
                                    logit2.append(1./2.*np.log((N_W[i]-lower_x)/(1.-N_W[i])))
                                    if bit==1:
                                        t_1_data.append(i)
                                        bit=0
                                if N_W[i]<lower_x:
                                    if not(not(logit2)):
                                        t_1_data.pop(-1)
                                        bit=1
                                    logit2=[]
                            else:
                                t_2_data.append(i-t_1_data[-1])
                                t_0=t_1_data[-1]
                                t_1_data[-1]=t_1_data[-1]-t_init[-1]
                                break

                        #logit2=np.array(logit2[:-1])

                        logit2=np.array(logit2)

                        w_MB=logit2.shape[0]

                        abs_2=np.arange(w_MB)

                        para_logit2,cov_logit2=opt.curve_fit(linear,abs_2,logit2,[-1.,0.])

                        h_MB, ord_MB = para_logit2

                        err=sum((logit2-(h_MB*abs_2+ord_MB))**2)/sum((logit2-np.mean(logit2))**2)

                        #MODIFICATION : Définition analytique de h

                        delta_x=upper_x-lower_x

                        h_bric=max(der_N)*delta_x/(val_ref-lower_x)/(upper_x-val_ref)

                        ord_bric=-h_bric*x_der-np.log((upper_x-val_ref)/(val_ref-lower_x))

                    if plot_logit==1:

                        length=len(N_W)
                        low1=[]
                        low2=[]
                        fake_data=[]
                        fake_data_2=[]

                        for i in range(length):
                            low1.append(x_out)
                            low2.append(lower_x)
                            fake_data.append(lower_x+delta_x/(1+np.exp(-h_bric*i-ord_bric)))

                        xdumb=np.arange(length)
                        
                        plt.plot(N_W)
                        plt.plot(fake_data,'o')
                        plt.plot(fake_data_2,'+')
                        plt.plot(xdumb,low1)
                        plt.plot(xdumb,low2)
                        plt.show()

                        yP3=linear(abs_2,h_MB,ord_MB)
                        yP4=linear(abs_2,h_bric,ord_MB)

                        plt.plot(abs_2, logit2, 'go', lw=2)
                        plt.plot(abs_2,yP3,'r',lw=2)
                        plt.plot(abs_2,yP4,'m',lw=2)
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
                        print h_MB
                        print h_analytic
                        print h_note
        

    if j%10==0:
        print j

    j_ctrl+=1




plt.plot(beta_data,phiII,'o')
plt.show()

plt.plot(delta_data,phiII,'o')
plt.show()

plt.plot(gamma_data,phiII,'o')
plt.show()

plt.plot(logphi,logP,'o')
plt.show()

plt.plot(t_1_data,'o')
plt.show()

plt.plot(t_2_data,'o')
plt.show()


logphi=np.array(logphi)
logP=np.array(logP)
para_P,cov_P=opt.curve_fit(linear,logphi,logP,[-1.,2.])
para_phi,covphi=opt.curve_fit(linear,logP,logphi,[-1.,2.])

print 'para_P = %s' %para_P
print 'para_phi = %s' %para_phi

alpa,betb=para_P[:]
yP=linear(logphi,alpa,betb)
plt.plot(logphi,yP)
plt.plot(logphi,logP,'o')
plt.show()

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

print coeff_P, coeff_phi, coeff_cross
print -coeff_phi/coeff_P,-coeff_cross/coeff_P

logphi=np.array(logphi)

yP2=linear(logphi,-coeff_phi/coeff_P,-coeff_cross/coeff_P)
p1, = plt.plot(logphi,yP2,label="Scaling law with exponent %.3f" % (-coeff_phi/coeff_P))
plt.plot(logphi,logP,'o')
plt.xlabel('log w ')
plt.ylabel('log h')
plt.title('Scaling law with delta =%f' %delta_min)
plt.legend(loc=1)
plt.show()


