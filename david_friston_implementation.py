import numpy as np
import matplotlib.pyplot as plt

#n:number of populations
n=25

a=40
b=40
A=3.25
B=22
c=135
c1=c 
c2=0.8*c
c3=0.25*c
c4=c3
r=0.56
e0=2.5
v0=6
h=0.001 #step size for RK

x1=0.1
x2=0.1
x3=0.1
v1=0.1
v2=0.1
v3=0.1
state_variables=np.array([x1,x2,x3,v1,v2,v3])
state_variables=np.repeat(state_variables,n, axis = 0).reshape(6,n)
state_variables=np.random.rand(6,n)


w= np.random.rand(n)+1/n
w=w/sum(w)


def S(var):
    return (2*e0)/(1+np.exp(r*(v0-(var))))
    
def f(x,s1,s2,s3):#derivatives
    return np.array([
    A * a * (p+c2*S(s1))-2*a*x[0,:]-x[3,:]*(a**2),
    B * b * c4*S(s2)-2*x[1,:]*b-x[4,:]*(b**2),
    A * a * S(s3)-2*a*x[2,:]-x[5,:]*(a**2),
    x[0,:],x[1,:],x[2,:]]) 


lastval=2 #total time in seconds
step_number=int(lastval/h)
y=np.zeros((step_number, n))
k=np.zeros(step_number)
for i in range(step_number):
    #input is gaussian dist with mu=220 and sigma=22
    p = np.random.normal(220, 22)
    s1 = state_variables[5,:].T @ w * c1
    s2 = state_variables[5,:].T @ w * c3
    s3 = (state_variables[3,:]-state_variables[4,:]).T @ w

    # parameters for RK2 :
    K1 = f(state_variables, s1, s2, s3)
    K2 = f(state_variables +  h * K1, s1, s2, s3)
    #output is y = v1-v2
    y[i,:]=(state_variables[3, :]-state_variables[4,:])/1000
    # new state variables formula:
    new_state_variables = state_variables + 0.5 * h * (K1+ K2)
    state_variables = new_state_variables


df_y=np.dot(y,w)

t=np.arange(0, lastval, h)
fft_y = np.fft.fft(y[:, 0])
freq = np.fft.fftfreq(len(y[:, 0]), h)


magnitude = np.abs(fft_y)
max_freq_idx = np.argmax(magnitude[1:]) + 1  # Exclude 0 frequency
max_freq = abs(freq[max_freq_idx])


t=np.arange(0, lastval, h)
plt.plot(t,df_y)
plt.xlabel("time [s]")
plt.title(f"Number of populations: {n} \nto_e = 1/a = {1000/a} ms   to_i = 1/b = {1000/b} ms\n max frequency greater than zero is {max_freq} Hz")



fft_df_y = np.fft.fft(df_y)
freq = np.fft.fftfreq(len(df_y), h)
plt.figure()
plt.plot(freq, np.abs(fft_df_y))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("FFT of David Friston y")
plt.show()

magnitude = np.abs(fft_df_y)
max_freq_idx = np.argmax(magnitude[1:]) + 1  # Exclude 0 frequency
max_freq = abs(freq[max_freq_idx])
print('max frequency greater than zero is ', max_freq, 'Hz')





