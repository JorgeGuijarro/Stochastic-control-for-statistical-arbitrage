import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad, simps
from numpy.random import multivariate_normal as gaussian
import time
import matplotlib.cm as cm

#Parameters
np.random.seed(1)

N = 100
A = np.diag(np.random.normal(loc = 0.5, scale = 0.1, size = N))  #eigenvalues of A are close so similar speeds of mean reversion in different components
mu = np.zeros(N)
sigma = np.random.uniform(low=-0.3,high=0.3, size = (N,N))
np.fill_diagonal(sigma, np.random.uniform(low=0,high=0.5,size=N))
inv = np.linalg.inv(sigma.dot(sigma.T))
p = np.ones(N)
X0 = mu

L = 400
T = 20  #mean(eigenvalues(A))*T is roughly the expected number of cycles
Deltat = T/L
M = 1000
r = 0.02
gamma = [1,2,3,4] #[1,2,3,4]
G = 4 #len(gamma)
alpha = [0,20,50] #[0,20,50]
Al = 1#len(alpha)
maxp = [1,2,4,8]
vectorp = [p+np.random.uniform(low=-i,high=i,size=N) for i in maxp]
lenP = len(vectorp)
Lambda = [0.1, 0.5,1]
Lam = len(Lambda)

X = np.zeros((L+1,N))
W = np.zeros((L+1,G,Al,1))#3))
pi = np.zeros((L+1,N,G,Al,1))#3))
WT = np.zeros((M,G,Al,1))#3))

colors = cm.rainbow(np.linspace(0, 1, G*Al))

#Functions

def Mu(x,Deltat):
    return expm(-A*Deltat).dot(x-mu) + mu

def integrand(s,Deltat):
    aux = expm(-A*(Deltat-s)).dot(sigma)
    return aux.dot(aux.T)

def Sigma(Deltat):
    integrated = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            integrated[i,j] = quad(lambda s: integrand(s,Deltat)[i,j], 0, Deltat)[0]
    return integrated

def exp_strat(x,alpha,gamma,r,t,p):
    P = p.reshape((N,1))
    summand1 = inv.dot(A.dot(mu-x)-p*r)
    temp = (A.dot(mu-x)-p*r)*(T-t)-A.dot(p)*r*(T-t)**2/2
    summand2 = A.T.dot(inv.dot(temp))
    return (summand1 + summand2)/(gamma*np.exp(r*(T-t)))

def meanvar_cont_strat(x,alpha,gamma,r,t,p):
    P = p.reshape((N,1))
    Inv = gamma*sigma.dot(sigma.T)+alpha*P.dot(P.T)
    return np.linalg.solve(Inv,(A.dot(mu-x)-p*r))*np.exp(r*(T-t))

def meanvar_disc_strat(x,alpha,gamma,r,l,p):
    P = p.reshape((N,1))
    Inv = gamma*Sigma0+alpha*P.dot(P.T)*Deltat
    b = ((np.eye(N)-expm(-A*Deltat)).dot(mu-x)-p*r*Deltat)*(1+r*Deltat)**(L-l-1)
    return np.linalg.solve(Inv,b)

def rate(t,gamma, Lambda):
    return np.sqrt(gamma/Lambda)*np.tanh(np.sqrt(gamma/Lambda)*(t-T))

'''def integrand_aim (x,gamma,r,t,p,Lambda,s):
    temp = np.exp(r*(T-s))*(A.dot(expm(A*(t-s)).dot(mu-x)-r*p)
    time_exp = np.cosh(np.sqrt(gamma/Lambda)*(s-T))/np.cosh(np.sqrt(gamma/Lambda)*(t-T)) #the time-ordered exponential is computable in closed for for this scalar rate
    return time_exp*temp'''

def integrand_aim_1 (gamma,r,t,Lambda,s):
    temp = np.exp(r*(T-s))*A.dot(expm(A*(t-s)))
    time_exp = np.cosh(np.sqrt(gamma/Lambda)*(s-T))/np.cosh(np.sqrt(gamma/Lambda)*(t-T))
    return time_exp*temp
                            
def integrand_aim_2 (gamma,r,t,Lambda,s):
    temp = np.exp(r*(T-s))
    time_exp = np.cosh(np.sqrt(gamma/Lambda)*(s-T))/np.cosh(np.sqrt(gamma/Lambda)*(t-T))
    return time_exp*temp

def aim_coefficients(gamma,r,t,Lambda):
    b1 = integrand_aim_1(gamma,r,t,Lambda,t)+integrand_aim_1(gamma,r,t,Lambda,T)
    dx = Deltat/4
    n = int((T-t)/dx)
    for i in range(1, n, 2):
        b1 += 4 * integrand_aim_1(gamma,r,t,Lambda,t + i * dx) #simpsons rule to approximate the integral                    
    for i in range(2, n-1, 2):
        b1 += 2 * integrand_aim_1(gamma,r,t,Lambda,t + i * dx)
    '''
    for i in range(N):
        for j in range(N):
            b1[i,j] = quad(lambda s: integrand_aim_1(gamma,r,t,p,Lambda,s)[i,j], t, T)[0]
    '''
    b2 = quad(lambda s: integrand_aim_2(gamma,r,t,Lambda,s), t, T)[0]
    
    return inv.dot(b1*dx/3)/Lambda, b2*r/Lambda

def save_aim_coefficients(gamma,r,L,Deltat,Lambda):
    for l in range(L):
        aim1,aim2 = aim_coefficients(gamma,r,l*Deltat,Lambda)
        np.save('aim1-'+str(l)+'.npy', b1)
        np.save('aim2-'+str(l)+'.npy', b2)
    return

def trans_cost_strat(X,index_gamma,r,l,p,index_Lambda):
    integrand = np.zeros((l+1,N))
    for v in range(l+1):
        time_exp = np.cosh(np.sqrt(gamma[index_gamma]/Lambda[index_Lambda])*(l*Deltat-T))/np.cosh(np.sqrt(gamma[index_gamma]/Lambda[index_Lambda])*(v*Deltat-T))
        aim = aims1[v,:,:,index_gamma,index_Lambda].dot(mu-X[v,:])- aims2[v,index_gamma,index_Lambda]*inv.dot(p)
        integrand[v,:] = time_exp*aim
    return np.trapz(integrand, axis = 0)

#######Simulations#####
'''
tic = time.time()
Sigma0 = Sigma(Deltat)
np.save('Sigma.npy', Sigma0)
'''
Sigma0 = np.load('Sigma.npy')
'''
time_elapsed = time.time() - tic
print('Time elapsed: ',time_elapsed, 'seconds') 
'''
X[0,:] = X0
'''
aims1 = np.zeros((L+1,N,N,G,Lam))
aims2 = np.zeros((L+1,G,Lam))

for l in range(L+1):
    for i in range(G):
        for j in range(Lam):
            aims1[l,:,:,i,j],aims2[l,i,j] = aim_coefficients(gamma[i],r,l*Deltat,Lambda[j])          

np.save('aims1.npy', aims1)
np.save('aims2.npy', aims2)
'''
'''
aims1 = np.load('aims1.npy')
aims2 = np.load('aims2.npy')                            

for m in range(M):
    for l in range(L):
        X[l+1,:] = gaussian(Mu(X[l,:],Deltat),Sigma0)
        for i in range(G):
            for j in range(Al):
                for k in range(1):#3):            
                    
                    W[l+1,i,j,k] = W[l,i,j,k]*(1+r*Deltat)+ pi[l,:,i,j,k].dot(X[l+1,:]-X[l,:]-r*p*Deltat)
                
                #pi[l+1,:,i,j,0] = exp_strat(X[l+1,:],alpha[j],gamma[i],r,Deltat*(l+1),p)
                #pi[l+1,:,i,j,1] = meanvar_cont_strat(X[l+1,:],alpha[j],gamma[i],r,Deltat*(l+1),p)
                #pi[l+1,:,i,j,2] = meanvar_disc_strat(X[l+1,:],alpha[j],gamma[i],r,Deltat*(l+1),p)
                
                #pi[l+1,:,i,j,0] = trans_cost_strat(X,i,r,l,p,j)
                
                    #W[l+1,i,j,k] = W[l,i,j,k]*(1+r*Deltat)+ pi[l,:,i,j,k].dot(X[l+1,:]-X[l,:]-r*vectorp[i]*Deltat)
                #pi[l+1,:,i,j,0] = exp_strat(X[l+1,:],alpha[j],gamma[0],r,Deltat*(l+1),vectorp[i])
                #pi[l+1,:,i,j,1] = meanvar_cont_strat(X[l+1,:],alpha[j],gamma[0],r,Deltat*(l+1),vectorp[i])
                #pi[l+1,:,i,j,2] = meanvar_disc_strat(X[l+1,:],alpha[j],gamma[0],r,Deltat*(l+1),vectorp[i])
                   
                pi[l+1,:,i,j,0] = trans_cost_strat(X,0,r,l,vectorp[i],2)
                            
    for i in range(G):
        for j in range(Al):
            for k in range(1):#3):
                WT[m,i,j,k] = W[L,i,j,k]
'''

####Plots###
#Plot of X
                
times = range(L+1)
for l in range(L):
    X[l+1,:] = gaussian(Mu(X[l,:],Deltat),Sigma0)
print(X)

plt.figure()
'''
plt.plot(times, X[:,0], c=colors[0])#, label = 'First coordinate')
plt.plot(times, X[:,1], c=colors[1])#, label = 'Second coordinate')
plt.plot(times, X[:,2], c=colors[2])#, label = 'Third coordinate')
plt.plot(times, X[:,3], c=colors[3])#, label = 'Fourth coordinate')
'''
plt.plot(times,X[:,:3])
#plt.title('Evolution of the PCA factors')
plt.xlabel('Time')
plt.ylabel('Xt')
#plt.legend()
plt.savefig('pathsX.png')
'''
for k in range(1):
    #Plots of a sample path of W
    
    #expectedW = W.sum(axis=0)/m
    
    
    for j in range(Al):#(Lam):
        plt.figure()
        for i in range(G):
            #plt.plot(times, W[:,i,j,k], c=colors[i*Al+j], label='gamma='+str(gamma[i])+',lambda='+str(Lambda[j]))
            plt.plot(times, W[:,i,j,k], c=colors[i*Al+j], label = 'p[i] = 1 + Unif(-'+str(maxp[i])+','+str(maxp[i])+')')
        plt.xlabel('Time')
        plt.ylabel('Wealth')
        plt.legend()
        plt.show()


    #Plots of a sample path of the first component of pi
    
    #expectedpi = pi.sum(axis=0)/m
    
    plt.figure()
    for j in range(Al):#(Lam):
        for i in range(G):
            #plt.plot(times, pi[:,0,i,j,k], c=colors[i*Al+j], label='gamma='+str(gamma[i])+',lambda='+str(Lambda[j]))
            plt.plot(times, pi[:,0,i,j,k], c=colors[i*Al+j], label = 'p[i] = 1 + Unif(-'+str(maxp[i])+','+str(maxp[i])+')')
        plt.xlabel('Time')
        plt.ylabel('Capital invested')
        plt.legend()
        plt.show()

    
    #Histograms of W_T

    #Colors = ['red', 'blue','green', 'yellow']
    #fig, axes = plt.subplots(nrows=2, ncols=2)
    #ax0, ax1, ax2, ax3 = axes.flatten()
    for j in range(Al):#(Lam):
        Colors = []
        for i in range(G):
            Colors += [colors[i*Al+j]]
        #labels = ['gamma=' +str(gamma[i])+',lambda=' + str(Lambda[j]) for i in range(G)]
        labels = ['p[i] = 1 + Unif(-'+str(maxp[i])+','+str(maxp[i])+')' for i in range(G)]
        Z=WT[:,:,j,k]
        plt.hist(Z, bins=60, normed=True, histtype='bar', stacked = True, color=Colors, label=labels)   
        plt.xlabel('Wealth')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

'''


            
