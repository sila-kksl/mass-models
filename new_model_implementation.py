import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#number of populations: n
n = 10

ae = 1.2
ai = 1
theta_e = 2.8
theta_i = 4
k = 1
r = 3
c22 = 28
c32 = 14
c23 = 28
c33 = 2
c11 = 4
c21 = 4
c12 = 2

#constants for frequency
tau_e = 0.05
tau_i = 0.05

w = np.random.rand(n) + 1/n
w = w/sum(w)


def sigmoid_e(x, theta_e):
    return 1 / (1 + np.exp(-ae*(x - theta_e)))

def sigmoid_i(x, theta_i):
    return 1 / (1 + np.exp(-ai*(x - theta_i)))

def system_of_equations(t, y):
    E1 = y[0]
    E2 = y[1]
    I1 = y[2]

    turev = np.zeros(3*n)
    for i in range(n):
        turev[0*n+i] = (-y[0*n+i] + (k + r * y[0*n+i]) * sigmoid_e(c11 * float(np.dot(y[0*n:1*n],w)) + c21 * float(np.dot(y[1*n:2*n],w)), theta_e )) * (1 / tau_e)
        turev[1*n+i] = (-y[1*n+i] + (k + r * y[1*n+i]) * sigmoid_e(c22 * float(np.dot(y[1*n:2*n],w)) + c12 * float(np.dot(y[0*n:1*n],w)) - c32 * float(np.dot(y[2*n:],w)), theta_e )) * (1 / tau_e)
        turev[2*n+i] = (-y[2*n+i] + (k + r * y[2*n+i]) * sigmoid_i(-c33 * float(np.dot(y[2*n:],w)) + c23 * float(np.dot(y[1*n:2*n],w)), theta_i )) * (1 / tau_i)
        
    return turev

time = np.linspace(0, 5, 1000)
initial_conditions = np.random.rand(3*n)*0.1
solution = solve_ivp(system_of_equations, (time[0], time[-1]), initial_conditions, t_eval=time)

#this solution matrix includes: 
#Dynamics of E1 at solution.y[0]
#Dynamics of E2 at solution.y[1]
#Dynamics of I1 at solution.y[2]


a = (solution.y[0*n:1*n] - solution.y[1*n:2*n])
av_y = np.dot(w, a)

#Dynamics of E1- E2 response modeling the effect of all 10 populations
plt.subplot(1, 2, 1)
plt.plot(solution.t, av_y, label="E1-E2 ", linewidth=2)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Neural Population Activity", fontsize=12)
plt.legend(fontsize=10)

plt.grid()

freq = np.fft.fftfreq(len(av_y), d=time[1] - time[0])
fft = np.fft.fft(av_y)
magnitude = np.abs(fft)
max_freq_idx = np.argmax(magnitude[1:]) + 1  # Exclude 0 frequency
max_freq = abs(freq[max_freq_idx])
print('maximum non-zero frequency component:', max_freq)

# Plot 2: FFT of E1-E2 response modeling the effect of all 10 populations
plt.subplot(1, 2, 2)
plt.plot(freq, np.abs(fft))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
#plt.title("FFT of David Friston y")
plt.suptitle(f"Dynamics of E1-E2\nNumber of population:{n}\nmaximum non-zero frequency component: {max_freq}", fontsize=14)
plt.tight_layout() 
plt.show()
