import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tools_HT

#----- Constantes ------

NDIM = 1

### Integration parameters
dt = 1
mu = 1e-3
g = 9.81
beta = 5e-2
t_final = 4000*dt
dx, dy = 0.01, 0.01
Nx, Ny = 1, 1


if NDIM == 1:

    ### Bar component
    rho = 2700  #density
    Cp = 900  # Thermal mass capacity
    Lambda = 22
    kappa = Lambda/(rho*Cp)

    ### Initialize class
    Tools_mat1D = tools_HT.Tools_matrix1D(0, 0)

    #-----------------------

    time = np.linspace(0, t_final, int(t_final/dt))
    indX = np.arange(0, Nx, dx)
    indY = np.arange(0, Ny, dy)
    X, Y = np.meshgrid(indX, indY)
    T1 = np.zeros(int(Nx/dx))
    T = np.zeros((len(time), len(T1)))

    T[0] = Tools_mat1D.initialize_bar(T.shape[1], dx, 100)
    Hx = Tools_mat1D.Hx_matrix1D(T.shape[1], dx)

    for i in range(len(time)-1):
        T[i+1] = Tools_mat1D.update(T[i], Hx, dt, kappa)
        T[i] = Tools_mat1D.border(T[i])

    ### Plots
    var = np.arange(0, Nx, dx)
    tools_HT.Plots(dt, Nx).Plot1D(var, T, time, name = 'xsquare1D')
    tools_HT.Plots(dt, Nx).anim1D(var, T, time, name = 0)

elif NDIM == 2:

    ### Bar component
    rho = 1e3  #density
    Cp = 4185  # Thermal mass capacity
    Lambda = 0.598
    kappa = Lambda/(rho*Cp)

    T_up = 0
    T_down = 100
    T_init = int(T_down/2)

    Tools_mat2D = tools_HT.Tools_matrix2D(T_init, T_up, T_down, kappa, dx, dy, dt, Lambda)
    time = np.linspace(0, t_final, int(t_final/dt))

    T1 = np.zeros(int(Nx/dx))
    T = np.zeros(((len(time), len(T1), len(T1))))
    Tools_mat2D.initialize_temp(T[0], up=True, down=True, init=True)

    Hx = Tools_mat2D.Hx_matrix2D(T.shape[1], dx)
    Hy = Tools_mat2D.Hx_matrix2D(T.shape[1], dy)

    for i in tqdm(range(len(time)-1)):
        T[i] = Tools_mat2D.initialize_temp(T[i], up=True, down=True, init=False)
        T[i, :, 0] = T_init
        T[i, :, -1] = T_init
        T[i+1] = Tools_mat2D.update2D(T, i, Hx, Hy)

    tools_HT.Plots(dt, Nx).Plot2D(T, name = 0)



else:
    raise TypeError("Enter 1 or 2 dimension..")
