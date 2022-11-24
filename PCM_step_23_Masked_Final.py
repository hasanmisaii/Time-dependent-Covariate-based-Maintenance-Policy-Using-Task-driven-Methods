# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:36:04 2021

@author: misaiiha

- Considering masked data
- Three differnt periods with random borders are considered
- The minimal, imperfect and perfect corrective maintenance actions with different costs are considered. 

Sixth Scenario: there are minimal, imperfect and perfect corrective maintenance actions with different costs
while I_1 and I_2 are Random considering Masked data
"""
import numpy as np
import pandas as pd
import itertools as ite
from scipy.stats import weibull_min
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.integrate import quad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
# from matplotlib import pyplot
from numpy import exp
from sklearn.neighbors import KernelDensity
import scipy.optimize as optimize
import math
from imblearn.over_sampling import SMOTE
import random

print("Hasan")
n = int(input("Enter number of data that should be generated ('n'):  \n"))  # number of samples

J = int(input("Enter number of system component(s): \n"))
arr_0 = input("Enter shape parameters by space: \n")   # takes the whole line of shape numbers by space
beta = list(map(float, arr_0.split(' ')))  # split those numbers with space( becomes ['2','3','6','6','5']) and then map every element into float (becomes [2,3,6,6,5])

arr_1 = input("Enter scale parameters by space: \n")   # takes the whole line of scale numbers by space
lam = list(map(float, arr_1.split(' ')))  # split those numbers with space( becomes ['2','3','6','6','5']) and then map every element into int (becomes [2,3,6,6,5])

# costs
cost = input("Enter costs for (Ins., Unavail.,MCM, ICM, PCM) by space: \n")
cost = list(map(float, cost.split(' ')))  # split those numbers with space( becomes ['2','3','6','6','5']) and then map every element into float (becomes [2,3,6,6,5])
cost_ins = cost[0]
cost_unavail = cost[1]
cost_mcm = cost[2]
cost_icm = cost[3]
cost_pcm = cost[4]
# %% failure cause
k_0 = list(ite.repeat(0, n))  # exact cause failure
k_1 = list(ite.repeat(0, n))  # Second cause failure
k_2 = list(ite.repeat(0, n))  # Third cause failure
t = list(ite.repeat(0, n))  # time to failure

t_0 = weibull_min.rvs(c=beta[0], loc=0, scale=lam[0], size=n)  # lifetimes of component 1
t_1 = weibull_min.rvs(c=beta[1], loc=0, scale=lam[1], size=n)  # lifetimes of component 2
t_2 = weibull_min.rvs(c=beta[2], loc=0, scale=lam[2], size=n)  # lifetimes of component 3

num_0I = 0
num_1I = 0
num_1M = 0
num_2I = 0
rho = 0.5

a_L1 = np.mean(beta)
a_L2 = np.mean(beta)
b_L1 = np.mean(lam)
b_L2 = np.mean(lam)

l_1 = weibull_min.rvs(c=a_L1, loc=0, scale=b_L1, size=1)
l_2 = 0
while l_2 < l_1:
    l_2 = weibull_min.rvs(c=a_L2, loc=0, scale=b_L2, size=1)/np.exp(-pow(l_1/b_L2, a_L2))


for i in range(n):
    t[i] = min(t_0[i], t_1[i], t_2[i])
    if i+1 >= n:
        if t[i] == t_0[i]:
            k_0[i] = "k_0"
        if t[i] == t_1[i]:
            k_0[i] = "k_1"
        if t[i] == t_2[i]:
            k_0[i] = "k_2"
        break
    if t[i] == t_0[i]:
        k_0[i] = "k_0"
        if t_0[i] > l_2:
            t_0[i+1] = weibull_min.rvs(beta[0], loc=0, scale=lam[0], size=1)
        if t_0[i] < l_1:
            t_0[i+1] = pow(-np.log(1-np.random.uniform(0, 1, 1)) * pow(lam[0], beta[0]) + pow(t_0[i], beta[0]), 1/beta[0]) - t_0[i]
        if l_1 < t_0[i] < l_2:
            num_0I += 1
            t_0[i+1] = pow(-np.log(1-np.random.uniform(0, 1, 1))/pow((1-rho), num_0I), 1/beta[0])*lam[0]
    else:
        t_0[i+1] = t_0[i]
    if t[i] == t_1[i]:
        k_0[i] = "k_1"
        if t_1[i] > l_2:
            t_1[i+1] = weibull_min.rvs(beta[0], loc=0, scale=lam[0], size=1)
        if t_1[i] < l_1:
            t_1[i+1] = pow(-np.log(1-np.random.uniform(0, 1, 1)) * pow(lam[0], beta[0]) + pow(t_0[i], beta[0]), 1/beta[0]) - t_0[i]
        if l_1 < t_1[i] < l_2:
            num_1I += 1
            t_1[i+1] = pow(-np.log(1-np.random.uniform(0, 1, 1))/pow((1-rho), num_0I), 1/beta[0])*lam[0]
    else:
        t_1[i+1] = t_1[i]
    if t[i] == t_2[i]:
        k_0[i] = "k_2"
        if t_2[i] > l_2:
            t_2[i+1] = weibull_min.rvs(beta[0], loc=0, scale=lam[0], size=1)
        if t_2[i] < l_1:
            t_2[i+1] = pow(-np.log(1-np.random.uniform(0, 1, 1)) * pow(lam[0], beta[0]) + pow(t_0[i], beta[0]), 1/beta[0]) - t_0[i]
        if l_1 < t_2[i] < l_2:
            num_0I += 1
            t_2[i+1] = pow(-np.log(1-np.random.uniform(0, 1, 1))/pow((1-rho), num_0I), 1/beta[0])*lam[0]
    else:
        t_2[i+1] = t_2[i]
# %% failure cause
k_1 = list(ite.repeat(0, n))  # Second cause failure
k_2 = list(ite.repeat(0, n))  # Third cause failure

for i in range(n):
    if t_0[i] < t_1[i] < t_2[i]:
        k_0[i] = "k_0"
        k_1[i] = "k_1"
        k_2[i] = "k_2"
    if t_2[i] < t_1[i] < t_0[i]:
        k_0[i] = "k_2"
        k_1[i] = "k_1"
        k_2[i] = "k_0"
    if t_1[i] < t_0[i] < t_2[i]:
        k_0[i] = "k_1"
        k_1[i] = "k_0"
        k_2[i] = "k_2"
    if t_1[i] < t_0[i] < t_2[i]:
        k_0[i] = "k_1"
        k_1[i] = "k_0"
        k_2[i] = "k_2"
    if t_2[i] < t_0[i] < t_1[i]:
        k_0[i] = "k_2"
        k_1[i] = "k_0"
        k_2[i] = "k_1"
    if t_0[i] < t_2[i] < t_2[i]:
        k_0[i] = "k_0"
        k_1[i] = "k_2"
        k_2[i] = "k_1"
    if t_1[i] < t_2[i] < t_0[i]:
        k_0[i] = "k_1"
        k_1[i] = "k_2"
        k_2[i] = "k_0"
# %% generate data as a data frame
data = {'time_to_failure': t,
        'cause_to_failure': k_0
        }
df = pd.DataFrame(data)
df['cause_to_failure'] = pd.Categorical(df.cause_to_failure)
print(df)
# visualization
print('The shape of our data is:', df.shape)
print('Describe all data:', df.describe(include='all'))
print('Describe categorical data:', df.describe(include=['category']))
print('Describe all data info:', df.info())
df.boxplot()
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
df.groupby('cause_to_failure').size().plot(kind='pie', autopct='%.2f', textprops={'fontsize': 20}, colors=['tomato', 'gold', 'skyblue'], ax=ax1)
ax1.set_ylabel('cause_to_failure', size=15)
plt.tight_layout()
plt.show()
# %% Real Cost Rate

# Pre-determination
K = 10
part = 100

M = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
MM = list(ite.repeat(0, K))
for i in range(K):
    MM[i] = random.sample(M, 1)[0]


def integrand_0(x, a, b):
    return x * weibull_min.pdf(x, c=a, loc=0, scale=b)


def integrand_1(x, a, b):
    return weibull_min.pdf(x, c=a, loc=0, scale=b)


def DL_1(x, a, b):
    return weibull_min.pdf(x, c=a, loc=0, scale=b)


def DL_2(x, a, b):
    return weibull_min.pdf(x, c=a, loc=0, scale=b)


def DL_12(x1, a, b):
    return weibull_min.pdf(x1, c=a, loc=0, scale=b) * quad(DL_2, x1, math.inf, args=(a, b))[0]


PL1L2 = quad(DL_12, 0, math.inf, args=(a_L2, b_L2))[0]


def long_run_cost_real(k, tau):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        a = beta[j]
        b = lam[j]
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = quad(integrand_0, (i)*tau, (i+1)*tau, args=(a, b))[0]
            R = 1
            for jj in range(J):
                if jj != j:
                    R *= quad(integrand_1, (i+1)*tau, math.inf, args=(a, b))[0]
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = quad(integrand_0, (h)*tau, (h+1)*tau, args=(a, b))[0]
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * quad(integrand_1, (h)*tau, (h+1)*tau, args=(a, b))[0] * R * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * quad(integrand_1, (h)*tau, (h+1)*tau, args=(a, b))[0] * R *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            # print(midpoint)
            # if midpoint <= tau:
            ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                quad(integrand_1, (i)*tau, (i+1)*tau, args=(a, b))[0] * R * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
            ex_time[j] += (i+1) * tau * quad(integrand_1, (i)*tau, (i+1)*tau, args=(a, b))[0] * R * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
        # print(ex_cost[j])
        # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_real_v = np.vectorize(long_run_cost_real)
long_run_cost_real_v(10, 0.2)

# tau_vals = np.linspace(5, 20, 10)
result_real = long_run_cost_real_v(K, tau_vals)
result_real

plt.plot(tau_vals, result_real)
plt.ylabel('OPTIMAL')
plt.show()
# %% RF for cause to failure
X = df[['time_to_failure']]
y = df[['cause_to_failure']]
# One-hot encode the data using pandas get_dummies
# X = pd.get_dummies(X)
# Display the first 5 rows of the last 12 columns
# X.iloc[:,5:].head(5)
# reshape data
# Imbanaced Classification
# transform the dataset
X, y = oversampler = SMOTE(k_neighbors=5).fit_resample(X, y)
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# define the model
model_cause_RF = RandomForestClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_cause_RF, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy RF with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification

# fit the model on the train dataset
model_cause_RF.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_RF.predict(X_test)
k_prob = model_cause_RF.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric RF
X_balanced = list(X['time_to_failure'])
y_balanced = list(y['cause_to_failure'])

data_balanced = {'time_to_failure': X_balanced,
                 'cause_to_failure': y_balanced
                 }
df_balanced = pd.DataFrame(data_balanced)
df_balanced['cause_to_failure'] = pd.Categorical(df_balanced.cause_to_failure)
print(df_balanced)


def Estimate_Par(t):
    def loglik(parms):
        S = 0
        a, b = parms
        for i in range(len(t)):
            S += np.log(weibull_min.pdf(t[i], c=a, loc=0, scale=b))
        return(-1*S)

    initial_guess = [1, 1]
    result = optimize.minimize(loglik, initial_guess, method='L-BFGS-B')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    return(fitted_params)


t_k_0 = np.array(df_balanced['time_to_failure'][df_balanced['cause_to_failure'] == 'k_0'])
t_k_1 = np.array(df_balanced['time_to_failure'][df_balanced['cause_to_failure'] == 'k_1'])
t_k_2 = np.array(df_balanced['time_to_failure'][df_balanced['cause_to_failure'] == 'k_2'])


def long_run_cost_estimate_PRF(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / part) * model_cause_RF.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * weibull_min.pdf(mid_0, c=a_0, loc=0, scale=b_0)
                    cycle_proba_0 += (tau / part) * model_cause_RF.predict_proba([[mid_0]])[0][j] *\
                        weibull_min.pdf(mid_0, c=a, loc=0, scale=b)
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PRF_v = np.vectorize(long_run_cost_estimate_PRF)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_PRF_v(K, 0.2, 50)

result_estimate_PRF = long_run_cost_estimate_PRF_v(K, tau_vals, part)
result_estimate_PRF

plt.plot(tau_vals, result_real)
plt.plot(tau_vals, result_estimate_PRF)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric RF
sample = np.asarray(t)
plt.hist(sample, bins=50, density=True)
plt.show()

model = KernelDensity(bandwidth=1, kernel='exponential')
sample = sample.reshape((len(sample), 1))
model.fit(sample)

...
# sample probabilities for a range of outcomes
values = np.asarray([value for value in range(0, len(sample))])
values = values.reshape((len(values), 1))
probabilities = model.score_samples(values)
probabilities = exp(probabilities)

...
# plot the histogram and pdf
plt.hist(sample, bins=50, density=True)
plt.plot(values[:], probabilities)
plt.show()


model = KernelDensity(bandwidth=1, kernel='exponential')
sample = X
model.fit(X)

model_0 = KernelDensity(bandwidth=1, kernel='exponential')


def long_run_cost_estimate_NPRF(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / part) * model_cause_RF.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * exp(model_0.score_samples([[mid_0]]))
                    cycle_proba_0 += (tau / part) * model_cause_RF.predict_proba([[mid_0]])[0][j] *\
                        exp(model.score_samples([[mid_0]]))
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPRF_v = np.vectorize(long_run_cost_estimate_NPRF)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_NPRF_v(K, 0.2, part)

result_estimate_NPRF = long_run_cost_estimate_NPRF_v(K, tau_vals, part)
result_estimate_NPRF

tau_vals[result_real == np.min(result_real)]
np.min(result_real)
tau_vals[result_estimate_NPRF == np.min(result_estimate_NPRF)]
np.min(result_estimate_NPRF)

plt.plot(tau_vals, result_estimate_PRF)
plt.plot(tau_vals, result_estimate_NPRF)
plt.plot(tau_vals, result_real)


plt.plot(tau_vals, result_real, color="b", ls='-')
plt.plot(tau_vals, result_estimate_NPRF, color="g", ls='--')
plt.legend(["Real", "RF"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
# %% KNN
# %% KNN for cause to failure
# define the model
model_cause_knn = KNeighborsClassifier(n_neighbors=5)
# evaluate the model
n_scores = cross_val_score(model_cause_knn, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy KNN with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_knn.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_knn.predict(X_test)
k_prob = model_cause_knn.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric KNN


def long_run_cost_estimate_PKNN(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / part) * model_cause_knn.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * weibull_min.pdf(mid_0, c=a_0, loc=0, scale=b_0)
                    cycle_proba_0 += (tau / part) * model_cause_knn.predict_proba([[mid_0]])[0][j] *\
                        weibull_min.pdf(mid_0, c=a, loc=0, scale=b)
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PKNN_v = np.vectorize(long_run_cost_estimate_PKNN)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_PKNN_v(K, 0.2, part)

result_estimate_PKNN = long_run_cost_estimate_PKNN_v(K, tau_vals, part)
result_estimate_PKNN


plt.plot(tau_vals, result_real)
plt.plot(tau_vals, result_estimate_PKNN)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric KNN


def long_run_cost_estimate_NPKNN(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / part) * model_cause_knn.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * exp(model_0.score_samples([[mid_0]]))
                    cycle_proba_0 += (tau / part) * model_cause_knn.predict_proba([[mid_0]])[0][j] *\
                        exp(model.score_samples([[mid_0]]))
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPKNN_v = np.vectorize(long_run_cost_estimate_NPKNN)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_NPKNN_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPKNN = long_run_cost_estimate_NPKNN_v(K, tau_vals, part)
result_estimate_NPKNN

tau_vals[result_real == np.min(result_real)]
np.min(result_real)
tau_vals[result_estimate_NPKNN == np.min(result_estimate_NPKNN)]
np.min(result_estimate_NPKNN)

plt.plot(tau_vals, result_estimate_PRF)
plt.plot(tau_vals, result_estimate_NPRF)
plt.plot(tau_vals, result_estimate_PKNN)
plt.plot(tau_vals, result_estimate_NPKNN)
plt.plot(tau_vals, result_real)


plt.plot(tau_vals, result_real, color="b", ls='-')
plt.plot(tau_vals, result_estimate_NPKNN, color="g", ls='--')
plt.legend(["Real", "KNN"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
# %% NB for cause to failure
# define the model
model_cause_gnb = GaussianNB()
# evaluate the model
n_scores = cross_val_score(model_cause_gnb, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy with CV NB: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_gnb.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_gnb.predict(X_test)
k_prob = model_cause_gnb.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric GNB


def long_run_cost_estimate_PGNB(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / part) * model_cause_gnb.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * weibull_min.pdf(mid_0, c=a_0, loc=0, scale=b_0)
                    cycle_proba_0 += (tau / part) * model_cause_gnb.predict_proba([[mid_0]])[0][j] *\
                        weibull_min.pdf(mid_0, c=a, loc=0, scale=b)
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PGNB_v = np.vectorize(long_run_cost_estimate_PGNB)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_PGNB_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PGNB = long_run_cost_estimate_PGNB_v(K, tau_vals, part)
result_estimate_PGNB

plt.plot(tau_vals, result_real)
plt.plot(tau_vals, result_estimate_PGNB)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric GNB


def long_run_cost_estimate_NPGNB(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / part) * model_cause_gnb.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * exp(model_0.score_samples([[mid_0]]))
                    cycle_proba_0 += (tau / part) * model_cause_gnb.predict_proba([[mid_0]])[0][j] *\
                        exp(model.score_samples([[mid_0]]))
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPGNB_v = np.vectorize(long_run_cost_estimate_NPGNB)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_NPGNB_v(K, 0.2, part)
# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPGNB = long_run_cost_estimate_NPGNB_v(K, tau_vals, part)
result_estimate_NPGNB

tau_vals[result_real == np.min(result_real)]
np.min(result_real)
tau_vals[result_estimate_NPGNB == np.min(result_estimate_NPGNB)]
np.min(result_estimate_NPGNB)

plt.plot(tau_vals, result_estimate_PGNB, color='c')
plt.plot(tau_vals, result_estimate_NPGNB, color='r')
plt.plot(tau_vals, result_real, color='b')


plt.plot(tau_vals, result_real, color="b", ls='-')
plt.plot(tau_vals, result_estimate_NPGNB, color="g", ls='--')
plt.legend(["Real", "NB"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
# %% SVM for cause to failure
# define the model
model_cause_svm = svm.SVC(probability=True)
# evaluate the model
n_scores = cross_val_score(model_cause_svm, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy SVM with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_svm.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_svm.predict(X_test)
k_prob = model_cause_svm.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)

print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric SVM


def long_run_cost_estimate_PSVM(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / part) * model_cause_svm.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * weibull_min.pdf(mid_0, c=a_0, loc=0, scale=b_0)
                    cycle_proba_0 += (tau / part) * model_cause_svm.predict_proba([[mid_0]])[0][j] *\
                        weibull_min.pdf(mid_0, c=a, loc=0, scale=b)
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PSVM_v = np.vectorize(long_run_cost_estimate_PSVM)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_PSVM_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PSVM = long_run_cost_estimate_PSVM_v(K, tau_vals, part)
result_estimate_PSVM


plt.plot(tau_vals, result_real)
plt.plot(tau_vals, result_estimate_PSVM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric SVM


def long_run_cost_estimate_NPSVM(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / part) * model_cause_svm.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * exp(model_0.score_samples([[mid_0]]))
                    cycle_proba_0 += (tau / part) * model_cause_svm.predict_proba([[mid_0]])[0][j] *\
                        exp(model.score_samples([[mid_0]]))
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPSVM_v = np.vectorize(long_run_cost_estimate_NPSVM)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_NPSVM_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPSVM = long_run_cost_estimate_NPSVM_v(K, tau_vals, part)
result_estimate_NPSVM

tau_vals[result_real == np.min(result_real)]
np.min(result_real)
tau_vals[result_estimate_NPSVM == np.min(result_estimate_NPSVM)]
np.min(result_estimate_NPSVM)


plt.plot(tau_vals, result_estimate_NPSVM)
plt.plot(tau_vals, result_estimate_PSVM)
plt.plot(tau_vals, result_real)


plt.plot(tau_vals, result_real, color="b", ls='-')
plt.plot(tau_vals, result_estimate_NPSVM, color="g", ls='--')
plt.legend(["Real", "SVM"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
# %% LDA
# %% LDA for cause to failure
# define the model
model_cause_lda = LinearDiscriminantAnalysis()
# evaluate the model
n_scores = cross_val_score(model_cause_lda, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy LDA with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_lda.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_lda.predict(X_test)
k_prob = model_cause_lda.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric LDA


def long_run_cost_estimate_PLDA(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / part) * model_cause_lda.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * weibull_min.pdf(mid_0, c=a_0, loc=0, scale=b_0)
                    cycle_proba_0 += (tau / part) * model_cause_lda.predict_proba([[mid_0]])[0][j] *\
                        weibull_min.pdf(mid_0, c=a, loc=0, scale=b)
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PLDA_v = np.vectorize(long_run_cost_estimate_PLDA)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_PLDA_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PLDA = long_run_cost_estimate_PLDA_v(K, tau_vals, part)
result_estimate_PLDA


plt.plot(tau_vals, result_real)
plt.plot(tau_vals, result_estimate_PLDA)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric LDA


def long_run_cost_estimate_NPLDA(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / part) * model_cause_lda.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * exp(model_0.score_samples([[mid_0]]))
                    cycle_proba_0 += (tau / part) * model_cause_lda.predict_proba([[mid_0]])[0][j] *\
                        exp(model.score_samples([[mid_0]]))
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPLDA_v = np.vectorize(long_run_cost_estimate_NPLDA)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_NPLDA_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPLDA = long_run_cost_estimate_NPLDA_v(K, tau_vals, part)
result_estimate_NPLDA

tau_vals[result_real == np.min(result_real)]
np.min(result_real)
tau_vals[result_estimate_NPLDA == np.min(result_estimate_NPLDA)]
np.min(result_estimate_NPLDA)

plt.plot(tau_vals, result_estimate_NPLDA)
plt.plot(tau_vals, result_estimate_PLDA)
plt.plot(tau_vals, result_real)

plt.plot(tau_vals, result_real, color="b", ls='-')
plt.plot(tau_vals, result_estimate_NPLDA, color="g", ls='--')
plt.legend(["Real", "LDA"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
# %% Decission Tree
# %% DT for cause to failure
# define the model
model_cause_dt = tree.DecisionTreeClassifier()
# evaluate the model
n_scores = cross_val_score(model_cause_dt, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy DT with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_dt.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_dt.predict(X_test)
k_prob = model_cause_dt.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)

print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric DT


def long_run_cost_estimate_PDT(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / part) * model_cause_dt.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * weibull_min.pdf(mid_0, c=a_0, loc=0, scale=b_0)
                    cycle_proba_0 += (tau / part) * model_cause_dt.predict_proba([[mid_0]])[0][j] *\
                        weibull_min.pdf(mid_0, c=a, loc=0, scale=b)
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PDT_v = np.vectorize(long_run_cost_estimate_PDT)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_PDT_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PDT = long_run_cost_estimate_PDT_v(K, tau_vals, part)
result_estimate_PDT


plt.plot(tau_vals, result_real)
plt.plot(tau_vals, result_estimate_PDT)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric DT


def long_run_cost_estimate_NPDT(k, tau, part):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, part + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / part) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / part) * model_cause_dt.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            ex_cost_0 = 0
            for h in range(i+1):
                midpoint_0 = 0
                cycle_proba_0 = 0
                p_0 = np.linspace(h*tau, (h+1)*tau, part + 1)
                n_0 = 1
                while n_0 <= (len(p_0)-1):
                    mid_0 = (p_0[n_0-1] + p_0[n_0])/2
                    midpoint_0 += (tau / part) * mid_0 * exp(model_0.score_samples([[mid_0]]))
                    cycle_proba_0 += (tau / part) * model_cause_dt.predict_proba([[mid_0]])[0][j] *\
                        exp(model.score_samples([[mid_0]]))
                    n_0 += 1
                ex_cost_0 += (((h+1) * tau - midpoint_0) * cost_unavail + cost_mcm) * cycle_proba_0 * (quad(DL_1, (h+1)*tau, math.inf, args=(a_L1, b_L1))[0]) +\
                    (((h+1) * tau - midpoint_0) * cost_unavail + cost_icm) * cycle_proba_0 *\
                    (quad(DL_1, 0, (h+1)*tau, args=(a_L1, b_L1))[0] * quad(DL_2, (h+1)*tau, math.inf, args=(a_L2, b_L2))[0])
            if midpoint <= tau:
                ex_cost[j] += ((mm) * cost_ins/J + ex_cost_0 + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                    cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                ex_time[j] += (i+1) * tau * cycle_proba * (quad(DL_2, 0, (i+1)*tau, args=(a_L2, b_L2))[0]/PL1L2)
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPDT_v = np.vectorize(long_run_cost_estimate_NPDT)
long_run_cost_real_v(K, 0.2)
long_run_cost_estimate_NPDT_v(K, 0.2, part)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPDT = long_run_cost_estimate_NPDT_v(K, tau_vals, part)
result_estimate_NPDT

tau_vals[result_real == np.min(result_real)]
np.min(result_real)
tau_vals[result_estimate_NPDT == np.min(result_estimate_NPDT)]
np.min(result_estimate_NPDT)

plt.plot(tau_vals, result_estimate_PDT)
plt.plot(tau_vals, result_estimate_NPDT)
plt.plot(tau_vals, result_real)


plt.plot(tau_vals, result_real, color="b", ls='-')
plt.plot(tau_vals, result_estimate_NPDT, color="g", ls='--')
plt.legend(["Real", "DT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')ha
# %% Parametric
plt.plot(tau_vals, result_real, color="b")
plt.axvline(x=tau_vals[result_real == np.min(result_real)], color="b")
plt.plot(tau_vals, result_estimate_PRF, linestyle=(1, (1, 1)), color="r")
plt.axvline(x=tau_vals[result_estimate_PRF == np.min(result_estimate_PRF)], linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_PSVM, linestyle=(2, (2, 1)), color="g")
plt.axvline(x=tau_vals[result_estimate_PSVM == np.min(result_estimate_PSVM)], linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_PKNN, linestyle=(3, (3, 1)), color="k")
plt.axvline(x=tau_vals[result_estimate_PKNN == np.min(result_estimate_PKNN)], linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_PGNB, linestyle=(4, (4, 1)), color="y")
plt.axvline(x=tau_vals[result_estimate_PGNB == np.min(result_estimate_PGNB)], linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_PLDA, linestyle=(5, (5, 1)), color="m")
plt.axvline(x=tau_vals[result_estimate_PLDA == np.min(result_estimate_PLDA)], linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_PDT, linestyle=(6, (6, 1)), color="c")
plt.axvline(x=tau_vals[result_estimate_PDT == np.min(result_estimate_PDT)], linestyle=(6, (6, 1)), color="c")
plt.legend(["Real", "PRF", "PSVM", "PKNN", "PNB", "PLDA", "PDT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
# %% Non Parametric
plt.plot(tau_vals, result_real, color="b")
plt.axvline(x=tau_vals[result_real == np.min(result_real)], color="b")
plt.plot(tau_vals, result_estimate_NPRF, linestyle=(1, (1, 1)), color="r")
plt.axvline(x=tau_vals[result_estimate_NPRF == np.min(result_estimate_NPRF)], linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_NPSVM, linestyle=(2, (2, 1)), color="g")
plt.axvline(x=tau_vals[result_estimate_NPSVM == np.min(result_estimate_NPSVM)], linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_NPKNN, linestyle=(3, (3, 1)), color="k")
plt.axvline(x=tau_vals[result_estimate_NPKNN == np.min(result_estimate_NPKNN)], linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_NPGNB, linestyle=(4, (4, 1)), color="y")
plt.axvline(x=tau_vals[result_estimate_NPGNB == np.min(result_estimate_NPGNB)], linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_NPLDA, linestyle=(5, (5, 1)), color="m")
plt.axvline(x=tau_vals[result_estimate_NPLDA == np.min(result_estimate_NPLDA)], linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_NPDT, linestyle=(6, (6, 1)), color="c")
plt.axvline(x=tau_vals[result_estimate_NPDT == np.min(result_estimate_NPDT)], linestyle=(6, (6, 1)), color="c")
plt.legend(["Real", "NPRF", "NPSVM", "NPKNN", "NPNB", "NPLDA", "NPDT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
# %% Parametric
plt.plot(tau_vals, result_real, color="b")
plt.plot(tau_vals, result_estimate_PRF, linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_PSVM, linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_PKNN, linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_PGNB, linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_PLDA, linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_PDT, linestyle=(6, (6, 1)), color="c")
plt.legend(["Real", "PRF", "PSVM", "PKNN", "PNB", "PLDA", "PDT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
# %% Non Parametric
plt.plot(tau_vals, result_real, color="b")
plt.plot(tau_vals, result_estimate_NPRF, linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_NPSVM, linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_NPKNN, linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_NPGNB, linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_NPLDA, linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_NPDT, linestyle=(6, (6, 1)), color="c")
plt.legend(["Real", "NPRF", "NPSVM", "NPKNN", "NPNB", "NPLDA", "NPDT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')

input("Press Enter to End:  \n")
