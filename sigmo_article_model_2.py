# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats

#PARAMETERS

M = 25000 #mémoire du système
time = 20000000

gamma_init = 0.23 #facteur d'implicature

np.random.seed(23)

#beta: steepness of P(x)
beta_min = 0.803
beta_max = 0.803

#delta: distance to criticality
delta_min = -8.
delta_max = -1.

loop = 1000 #number of processes

size = 2.0 #As the parameter size increases, the size of the relevant time window on which produced occurrences are counted also increases.
M_W = int(M*size)
W = 5 #Number of time windows on which the sliding average is computed.

tol_c = 10 ** (-8)
time_spent_critical = 10000

#Growth tracking parameters

a_k = 11 #number of additional decades on each side of the time range selected on which the algorithm tries to optimize the logit.

min_nb = 6 #minimal number of points for the logit

min_nb_growth = 2 #minimal number of consecutive frequency increase

tol = 2 #number of successive decades allow to decrease in growth phase or to increase in latency

min_growth = 0.10 #threshold of the relative derivative above which the growth is assumed to be significative

good_fit = 0.98 #minimal r^2 value above which the fit is considered to be good

stagnancy_tol = 0.15 #size of the maximal step in frequency tolerated by the latancy determination

tol_lat = 5 #number of successive decades during which a null frequency is assumed to be indicative of latency

dominance_threshold = 0.9 #value above which the last point of a pair should be for the pair to be considered a proper candidate


##OPTIONS

plot_logit=0 #=1 to plot the logit transformation of each process.

para_fixed=0 #=1 to fix the value of the parameters for all loops

alloc=0 #=1 to consider an alternative mechanism

both_mechs=0 #=1 # to consider both speaker/producer and hearer/interpreter mechanism

display=1 #=1 to display information on each process

#INITIALIZING LISTS AND INDICES

j_loop = 0

##phiII = [0] * time
##pente = [0] * time
##phiI = [0] * time
##logphi = [0] * time
##logP = [0] * time
##beta_data = [0] * time
##gamma_data = [0] * time
##delta_data = [0] * time
##err_data = []
##total_time_data = [0] * time

phiII = []
pente = []
phiI = []
logphi = []
logP = []
beta_data = []
gamma_data = []
delta_data = []
err_data = []
total_time_data = []

##DEFINITIONS DE FONCTIONS##

def alpha(x):
    return (np.arctanh(2. * x - 1)) ** 2

def linear(x,alp,bet):
    return alp * x + bet

def balance(x,gam,beta):
    return 1. / 2. * (1 + np.tanh(beta * ((x + gam) - (1 - x)) / np.sqrt((x + gam) * (1 - x)))) - x

def balance_der(x,gam,beta):
    return 1./4. * beta * (1 + gam) ** 2 * np.exp(- 3. / 2. * np.log((x + gam) * (1 - x))) * (1 - (np.tanh(beta * ((x + gam) - (1 - x)) / np.sqrt((x + gam) * (1 - x)))) ** 2) - 1

def critical_point(beta,gamma_init,ctrl_max,tol_crit,bm_value):
    g = gamma_init
    coeff_x = [1, 3 * g-1, beta ** 2 * (1 + g) ** 4 + 3 * g * (g - 1), (g - 3) * g ** 2, - g ** 3]
    r = np.roots(coeff_x)
    for i in range(len(r)):
        if r[i] == np.conjugate(r[i]):
            r[i] = r[i].real
            if (r[i] > 0) & (r[i] < 1):
                a = r[i]
                a = a.real
                break
    ctrl = 0
    db = np.abs(balance(a,g,beta))
    dder = np.abs(balance_der(a,g,beta))   
    while (db > tol_crit) | (dder > tol_crit):
        if ctrl < ctrl_max: ##Une condition servant à éviter d'être coincé indéfiniment au cas où la récursion ne fonctionnerait pas.            
            coeff_g = [beta ** 2, 2 * beta ** 2*(2 * a - 1) - (1 - a) * alpha(a), beta ** 2 * (2 * a - 1) ** 2 - a * (1 - a) * alpha(a)]         
            r_g = np.roots(coeff_g)        
            for i in range(len(r_g)):
                if r_g[i] == np.conjugate(r_g[i]):
                    r_g[i] = r_g[i].real
                    if (r_g[i] > 0) & (r_g[i] < 1):
                        g = r_g[i]
                        g = g.real
                        break
            coeff_x = [1, 3 * g - 1, beta ** 2 * (1+g) ** 4 + 3 * g * (g - 1), (g - 3) * g ** 2, - g ** 3]
            r = np.roots(coeff_x)
            for i in range(len(r)):
                if r[i] == np.conjugate(r[i]):
                    r[i] = r[i].real
                    if (r[i] > 0) & (r[i] < 1):
                        a = r[i]
                        a = a.real
                        break
            db = np.abs(balance(a,g,beta))
            dder = np.abs(balance_der(a,g,beta))
            g_look = g
            ctrl += 1
        else:
            break
    if both_mechs == 1:
        g = np.sqrt(1 + g) - 1

    return g, db, dder

def random_walk(gamma,time,W,alloc,both_mechs):
    memoire = [0] * M
    N = 0
    x = float(N) / float(M)
    N_tot = []
    N_M = 0
    site_call = 0
    count = 0
    countdown = 0

    for i in xrange(time):
      
        if (alloc == 0) & (both_mechs == 0):
            coin_site=1.
        else:
            coin_site = np.random.uniform(0,1)
            coin_add = np.random.uniform(0,1)

        if coin_site > 0.5:
            site_call += 1
            if (alloc == 1) & (both_mechs == 0):
                f = x
            else:
                f = (x + gamma) / (1 + gamma)
            if f != 0.0:
                if f != 1.:
                    phi = (2 * f - 1) / np.sqrt(f * (1 - f))
                    P = 1. / 2. * (1 + np.tanh(beta * phi))
                else:
                    P = 1.0
            else:
                P = 0.0
            dice = np.random.uniform(0,1)
            if P > dice:
                N += 1
                memoire.append(1)
                N_M += 1
            else:
                memoire.append(0)
            erased = np.random.randint(0,M)
            occ = memoire[erased]
            if occ == 1:
                N -= 1
            memoire.remove(occ)
            x = float(N) / float(M)
        else :
            if coin_add < gamma:
                N += 1
                memoire.append(1)
                erased = np.random.randint(0,M)
                occ = memoire[erased]
                if occ == 1:
                    N -= 1
                memoire.remove(occ)
                x = float(N) / float(M)

        if i != 0:
            if i % M_W == 0:
                x_occ = float(N_M) / float(site_call)
                N_tot.append(x_occ)
                N_M = 0
                site_call = 0
                if (x_occ > (1. - 1. / float(M))):
                    count = 1
                if count == 1:
                    countdown += 1
                if countdown > W + 1:
                        time_end = i
                        break

    N_W = []

    for k in range(len(N_tot) - (W - 1)):
        N_W.append((sum(N_tot[k:k+W])) / float(W))

    return N_W

        


##DEBUT DE LA BOUCLE##

init = 0

while j_loop < loop:

    beta = np.random.uniform(beta_min,beta_max)

    delta_gamma_power = np.random.uniform(delta_min,delta_max)

    delta_gamma = 10 ** delta_gamma_power


    ##DETERMINATION DU GAMMA

    if init == 0:
        g_c, db, dder = critical_point(beta,gamma_init,time_spent_critical,tol_c,both_mechs)
        g_fixed = g_c
        delta_fixed = delta_gamma
        beta_fixed = beta
        init = 1
        
    if para_fixed == 1:
        g_c = g_fixed
        delta_gamma = delta_fixed
        beta = beta_fixed

    else:
        g_c, db, dder = critical_point(beta,gamma_init,time_spent_critical,tol_c,both_mechs)

    print g_c
        
    gamma = g_c * (1 + delta_gamma)

    if (db < 10 ** (- 6))&(dder < 10 ** (- 6)):

        #Marche aléatoire

        N_W = random_walk(gamma,time,W,alloc,both_mechs)

        Nmax=max(N_W)

    # TRAITEMENT DES RESULTATS DE LA MARCHE ALEATOIRE
        
        if Nmax > (1. - 10./float(M)):

            relative_der = []

            relative_der = [ (N_W[i + 1] - N_W[i - 1]) / (2.0 * N_W[i]) for i in range(1,len(N_W) - 1)]
            
            relative_der = [(N_W[1] - N_W[0]) / N_W[0]] + relative_der + [(N_W[len(N_W) - 1] - N_W[len(N_W) - 2]) / N_W[len(N_W) - 1]]
            
            growth = []
            
            for k in range(len(relative_der)):
                if relative_der[k] > min_growth:
                    growth.append(k)
            growth_parts = []
            growth_part = []
            while growth != []:
                if growth_part != []:
                    if growth[0] - growth_part[-1] <= 1 + tol:
                        growth_part.append(growth[0])
                        growth.pop(0)
                    else:
                        growth_parts.append(growth_part)
                        growth_part= []
                else:
                    growth_part.append(growth[0])
                    growth.pop(0)
            growth_parts.append(growth_part)

            growth_parts = [p for p in growth_parts if len(p) >= min_nb_growth]

            if growth_parts == []:
                if display == 1:
                    print 'No growth phase detected.'
                err_data.append(1)
            else:

                for part in range(len(growth_parts)):
                    pairs = []
                    r_values = []
                    slopes = []
                    intercepts = []
                    born_1 = max(growth_parts[part][0] - a_k, 0)
                    born_2 = min(growth_parts[part][-1] + a_k,len(N_W) - 1)
                    for i_min in range(born_1,born_2 - min_nb):
                        for i_max in range(i_min + min_nb - 1,born_2):
                            if all(N_W[k] < N_W[i_max] for k in range(i_min, i_max)) & all(N_W[k] > N_W[i_min] for k in range(i_min + 1, i_max)):
                                if N_W[i_max] > dominance_threshold:
                                    pairs.append([i_min,i_max])
                                    logit = [np.log((N_W[i] - N_W[i_min]) / (N_W[i_max] - N_W[i])) for i in range(i_min + 1,i_max)]
                                    x = np.arange(len(logit))
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,logit)
                                    r_values.append(r_value ** 2)
                                    slopes.append(slope)
                                    intercepts.append(intercept)

                                
                good_pairs = [pairs[j] for j in range(len(pairs)) if r_values[j] > good_fit]
                if good_pairs == []:
                    if display == 1:
                        if r_values != []:
                            print 'The linear fit of the logit is rather poor : %s' %max(r_values)
                            print '***'
                        else:
                            print 'There is no growth period long enough for logit analysis.'
                    err_data.append(1)
                else:
                    good_rs = [r_values[j] for j in range(len(r_values)) if r_values[j] > good_fit]
                    width = [good_pairs[j][1] - good_pairs[j][0] for j in range(len(good_pairs))]
                    r_left = [good_rs[j] for j in range(len(width)) if width[j] == max(width)]
                    indices = [j for j in range(len(width)) if width[j] == max(width)]
                    selected_pair = good_pairs[indices[r_left.index(max(r_left))]]
                    i_min, i_max = selected_pair[0], selected_pair[1]
                    slope = slopes[pairs.index(selected_pair)]
                    intercept = intercepts[pairs.index(selected_pair)]
                    logit = [np.log((N_W[i] - N_W[i_min]) / (N_W[i_max] - N_W[i])) for i in range(i_min + 1,i_max)]
                    if display == 1:
                        print 'Linear fit successful'
                        print 'r² = %s' %max(r_left)
                        print 'slope = %.2f' %slope
                    err_data.append(0)

##                    beta_data[j_loop] = beta
##                    gamma_data[j_loop] = gamma
##                    delta_data[j_loop] = delta_gamma
##                    phiII[j_loop] = len(logit) + 2
##                    pente[j_loop] = slope
##                    logphi[j_loop] = np.log(len(logit) + 2)
##                    logP[j_loop] = np.log(slope)
##                    total_time_data[j_loop] = len(N_W)

                    beta_data.append(beta)
                    gamma_data.append(gamma)
                    delta_data.append(delta_gamma)
                    phiII.append(len(logit) + 2)
                    pente.append(slope)
                    logphi.append(np.log(len(logit) + 2))
                    logP.append(np.log(slope))
                    total_time_data.append(len(N_W))

                    j_loop += 1

                
                    stagnancy = []
                    for i in range(1,selected_pair[0]):
                        i_start = selected_pair[0]
                        diff = N_W[selected_pair[1]] - N_W[selected_pair[0]]
                        freq_ref = N_W[selected_pair[0]]
                        i_a = i_start - i  
                        i_count = i_a
                        bit_zero = 0
                        if (N_W[i_a] < freq_ref + diff * stagnancy_tol) & (N_W[i_a] > freq_ref - diff * stagnancy_tol) :
                            stagnancy.append(i_a)
                        else:
                            break
                        stagnancy = [stagnancy[k] for k in range(len(stagnancy)) if N_W[stagnancy[k]] > 0.0] 
                    if stagnancy != []:
                        if stagnancy[0] >= selected_pair[0] - tol :
                            for j in range(len(stagnancy)):
                                if j == len(stagnancy) - 1:
                                    starting_point = stagnancy[j]
                                else:
                                    if stagnancy[j + 1] < stagnancy[j] - tol_lat:
                                        starting_point = stagnancy[j]
                                        break
                        else:
                            if display == 1:
                                print 'There is no latency'
                            starting_point = selected_pair[0]
                    else:
                        if display == 1:
                            print 'There is no latency'
                        starting_point = selected_pair[0]
                    phiI.append(selected_pair[0] - starting_point)

                    if plot_logit == 1: 
                        lin_logit = [slope * i + intercept for i in range(-1,len(logit)+1)]
                        absc_i = [i for i in range(-1, len(logit)+1)]
                        plt.plot(logit, 'ro', label='Data points')
                        plt.plot(absc_i,lin_logit, 'g-', label='Linear fit')
                        plt.axis([-0.2, len(logit) - 0.8, min(logit) - 0.2, max(logit) + 0.2])
                        plt.title('Logit of the sigmoidal growth')
                        plt.legend(loc=2)
                        plt.show()

                        
                        plt.plot(range(starting_point),N_W[0:starting_point],'g+')
                        plt.plot(range(starting_point,selected_pair[0]),N_W[starting_point:selected_pair[0]],'ro')
                        plt.plot(range(selected_pair[0],min(selected_pair[1]+1,len(N_W))),N_W[selected_pair[0]:min(selected_pair[1]+1,len(N_W))],'b*')
                        plt.plot(range(min(selected_pair[1]+1,len(N_W)),len(N_W)),N_W[min(selected_pair[1]+1,len(N_W)):],'mx')
                        plt.show()
                        
        else:
            if display == 1:
                print 'Replacement did not happen.'
            err_data.append(1)

    
    else:
        if display == 1:
            print 'Critical point not found.'
        err_data.append(1)
                
     
    if j_loop % 10 == 0:
        print j_loop



inverse_time = [1. / total_time_data[i] for i in range(len(total_time_data))]

plt.plot(beta_data,phiII,'o')
plt.title('Time of growth phase vs beta parameter')
plt.show()

plt.plot(np.log10(delta_data),phiII,'o')
plt.title('Time of growth phase vs delta paramater - semilog plot')
plt.show()

plt.plot(gamma_data,phiII,'o')
plt.title('Time of growth phase vs gamma parameter')
plt.show()

plt.plot(np.log10(delta_data),inverse_time,'o')
plt.title('Inverse time of replacement vs delta parameter -semilog plot')
plt.show()

plt.plot(gamma_data,inverse_time,'o')
plt.title('Inverse time of repalcement vs gamma parameter')
plt.show()

plt.bar(list(set(phiI)),[phiI.count(list(set(phiI))[i]) for i in range(len(list(set(phiI))))])
plt.title('Distribution of latency times')
plt.show()

plt.bar(list(set(phiII)),[phiII.count(list(set(phiII))[i]) for i in range(len(list(set(phiII))))])
plt.title('Distribution of growth time')
plt.show()


logphi=np.array(logphi)
logP=np.array(logP)

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


