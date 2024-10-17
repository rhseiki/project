import numpy as np

# Problem 1

# Funtion: f(x) = xsin(3x) - exp(x)

# Newton-Raphson method

# Derivative: f'(x) = sin(3x) + 3xcos(x) - exp(x)
# Initial guess: x = -1.6

x = np.array([-1.6]); # Initial guess
for j in range(1000):
    x = np.append(x, x[j] - (x[j] * np.sin(3 * x[j]) - np.exp(x[j]))/(np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])));
    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j]);
    if abs(fc) < 1e-6:
        break
A1 = x
NR_iterations = j + 2;

# Bisection method

xl = -0.7; xr = -0.4; # Initialize the end points
x_mid = []; # Initialization of mid values
for j in range(1000): 
    xc = (xl + xr) / 2;   # Define the midpoint
    x_mid = np.append(x_mid, xc);
    fc = xc * np.sin(3 * xc) - np.exp(xc);
    if ( fc > 0 ):
        xl = xc;
    else:
        xr = xc;
    if ( abs(fc) < 1e-6 ):
        break

A2 = x_mid
Bij_iterations = j + 1;

A3 = np.array([NR_iterations, Bij_iterations])


# Problem 2

A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1 , 0])
y = np.array([0 , 1])
z = np.array([1 , 2, -1])

A4 = A + B

A5 = 3*x - 4*y

A6 = A @ x

A7 = B @ (x-y)

A8 = D @ x

A9 = D @ y + z

A10 = A @ B

A11 = B @ C

A12 = C @ D