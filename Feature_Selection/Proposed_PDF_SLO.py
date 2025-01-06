import numpy as np
import copy
import numpy.random as rnd
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Training import X_snow_PDF
import matplotlib.pyplot as plt
import sys
import cv2
import math
Y_min = float('inf')
Y_max = -float('inf')
Xw, Yw, Zw = [0.95, 1.0, 1.09]
uw = (4 * Xw) / (Xw + (15 * Yw) + (3 * Zw))
vw = (9 * Yw) / (Xw + (15 * Yw) + (3 * Zw))
A = 0
B = 1
# To get the minimum and maximum no of iteration
for i in range(100, 100 + 1):
    for j in range(100,100 + 1):
        b, g, r = 345,456,787
        if b > 255:
            b = 255
        elif b < 0:
            b = 0
        if g > 255:
            g = 255
        elif g < 0:
            g = 0
        if r > 255:
            r = 255
        elif r < 0:
            r = 0

        b = b / 255
        g = g / 255
        r = r / 255

        if (r < 0.03928):
            r = r / 12.92
        else:
            r = pow(((r + 0.055) / 1.055), 2.4)
        if (g < 0.03928):
            g = g / 12.92
        else:
            g = pow(((g + 0.055) / 1.055), 2.4)
        if (b < 0.03928):
            b = b / 12.92
        else:
            b = pow(((b + 0.055) / 1.055), 2.4)
        # XYZ values
        X = ((0.412453 * r) + (0.35758 * g) + (0.180423 * b))
        Y = ((0.212671 * r) + (0.71516 * g) + (0.072169 * b))
        Z = ((0.019334 * r) + (0.119193 * g) + (0.950227 * b))
        # Maximum and Minimum values of Y
        if (Y > Y_max):
            Y_max = Y
        if (Y < Y_min):
            Y_min = Y
#In population‐based optimization algorithms, population members are identified using a matrix called the population matrix.
for i in range(200, 200 + 1):
    for j in range(200, 200+ 1):
        b, g, r = 435,546,675

        if b > 255:
            b = 255
        elif b < 0:
            b = 0
        if g > 255:
            g = 255
        elif g < 0:
            g = 0
        if r > 255:
            r = 255
        elif r < 0:
            r = 0
        # Travel Routes and Movement
        b = b / 255
        g = g / 255
        r = r / 255

        # Hunting
        if (r < 0.03928):
            r = r / 12.92
        else:
            r = pow(((r + 0.055) / 1.055), 2.4)
        if (g < 0.03928):
            g = g / 12.92
        else:
            g = pow(((g + 0.055) / 1.055), 2.4)
        if (b < 0.03928):
            b = b / 12.92
        else:
            b = pow(((b + 0.055) / 1.055), 2.4)

        # Reproduction
        X = ((0.412453 * r) + (0.35758 * g) + (0.180423 * b))
        Y = ((0.212671 * r) + (0.71516 * g) + (0.072169 * b))
        Z = ((0.019334 * r) + (0.119193 * g) + (0.950227 * b))

        # Mortality
        if X == 0 and Y == 0 and Z == 0:
            x = 0.3127
            y = 0.3291
        else:
            x = X / (X + Y + Z)
            y = Y / (X + Y + Z)

        # Calculate location of prey 𝑝௜,ௗ using Equation (6)
        if Y > Y_max:
            Y = 1
        elif Y < Y_min:
            Y = 0
        else:
            Y = 0

        # Calculate 𝑥௜௉ଶ ,ௗ using Equation (7).
        if y == 0:
            X = 0
            Y = 0
            Z = 0
        else:
            X = x * Y / y
            Y = Y
            Z = (1 - x - y) * Y / y

        r = ((3.240479 * X) + (-1.53715 * Y) + (-0.498535 * Z))
        g = ((-0.969256 * X) + (1.875991 * Y) + (0.041556 * Z))
        b = ((0.055648 * X) + (-0.204043 * Y) + (1.057311 * Z))

        if (r < 0.00304):
            r = 12.92 * r
        else:
            r = (1.055 * pow(r, (1 / 2.4))) - 0.055
            if (r > 1):
                r = 1
        if (g < 0.00304):
            g = 12.92 * g
        else:
            g = (1.055 * pow(g, (1 / 2.4))) - 0.055
            if (g > 1):
                g = 1
        if (b < 0.00304):
            b = 12.92 * b
        else:
            b = (1.055 * pow(b, (1 / 2.4))) - 0.055
            if (b > 1):
                b = 1

        if math.isnan(r):
            r = 1
        if math.isnan(g):
            g = 1
        if math.isnan(b):
            b = 1
        r = int(r * 255 + 0.5)
        g = int(g * 255 + 0.5)
        b = int(b * 255 + 0.5)

def displacement(X):

    f = sum(100.0 * (X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2 for i in range(0, len(X) - 1))
    return f

def hunting(X):

    f_sum = sum((X[i] ** 4) - (16 * X[i] ** 2) + (5 * X[i]) for i in range(len(X)))
    return f_sum / 2

def reproduction(x):

    d = len(x)
    a = 20
    b = 0.2
    c = np.pi * 2
    sum1 = sum(x[i] ** 2 for i in range(d))
    sum1 = (-a) * np.exp(((-b) * np.sqrt(sum1 / d)))
    sum2 = sum(np.cos(c * x[i]) for i in range(d))
    sum2 = np.exp((sum2 / d))
    return sum1 - sum2 + a + np.exp(1)


def local_best_get(particle_pos, particle_pos_val, p):
    local_best = [0] * p  # creating empty local best list
    for j in range(p):  # finding the best snow in each neighbourhood
        # and storing it in 'local_best'
        local_vals = np.zeros(3)
        local_vals[0] = particle_pos_val[j - 2]
        local_vals[1] = particle_pos_val[j - 1]
        local_vals[2] = particle_pos_val[j]
        min_index = int(np.argmin(local_vals))
        local_best[j - 1] = particle_pos[min_index + j - 2][:]
    return np.array(local_best)


def initiation(f, bounds, p):

    d = len(bounds)  # finding number of dimensions
    particle_pos = np.zeros(p)  # creating empty position array
    particle_pos = particle_pos.tolist()  # converting array to list
    snow_velocity = particle_pos[:]  # empty array
    particle_pos_val = particle_pos[:]  # empty value array
    for j in range(p):  # iterating ovre the number of snow
        particle_pos[j] = [rnd.uniform(bounds[i][0], bounds[i][1]) \
                           for i in range(d)]  # random coordinate within bounds
        particle_pos_val[j] = f(particle_pos[j])  # calculating function value
        # at each snow
        snow_velocity[j] = [rnd.uniform(-abs(bounds[i][1] - bounds[i][0]) \
                                            , abs(bounds[i][1] - bounds[i][0])) for i in range(d)]
        # creating PDF values for each dimension
    lmbda = 2
    # Probability values
    poisson_pd = poisson.pmf(X_snow_PDF, lmbda)
    print("selects the leopard for guiding in the row based value is: ",poisson_pd)
    local_best = local_best_get(particle_pos, particle_pos_val, p)
    snow_best = particle_pos[np.argmin(particle_pos_val)]  # getting the lowest snow value
    particle_best = copy.deepcopy(particle_pos)  # setting all snow current positions to best
    return d, np.array(particle_pos), np.array(particle_best), \
           np.array(snow_best), np.array(snow_velocity), np.array(local_best), \
           np.array(particle_pos_val)

def withinbounds(bounds, particle_pos):

    for i in range(len(bounds)):
        if particle_pos[i] < bounds[i][0]:  # if  p‐value of less than 0.05
            particle_pos[i] = bounds[i][0]
        elif particle_pos[i] > bounds[i][1]:  # if  p‐value of higher than 0.05
            particle_pos[i] = bounds[i][1]
    return
