import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tools_HT
from matplotlib.animation import FuncAnimation, PillowWriter

#----- Constantes ------
dt = 0.1
mu = 1e-3
g = 9.81
beta = 5e-2
t_final = 3000*dt
dx, dy = 0.01, 0.01
Nx, Ny = 1, 1
T_init = 0
T_down = 20
T_up = -20
T_side = 0
rho = 2.7e3  #density
Cp = 0.888e3  # Thermal mass capacity
Lambda = 237
kappa = Lambda/(rho*Cp)
Tools = tools_HT.Tools(T_init, T_down, T_up, T_side, kappa, dt, dx, dy, mu, rho, g, beta)
Tools_mat1D = tools_HT.Tools_matrix1D(0, 100, 100)

#-----------------------

time = np.linspace(0, t_final, int(t_final/dt))
indX = np.arange(0, Nx, dx)
indY = np.arange(0, Ny, dy)
X, Y = np.meshgrid(indX, indY)
T1 = np.zeros(int(Nx/dx))
T = np.zeros((len(time), len(T1)))

Tools.print_constante(T1, t_final)
T[0] = Tools_mat1D.initialize_bar(T.shape[1], dx, 100)

Hx = Tools_mat1D.Hx_matrix1D(T.shape[1], dx)


for i in range(len(time)-1):
    #T[i+1] = Tools_mat1D.initialize_temp(T[i], left = True, right = True, init = False)
    T[i+1] = T[i] - dt * kappa*Hx@T[i]
    T[i+1, 0] = 0
    T[i+1, -1] = 0


plt.figure()
t=0
ind = np.linspace(0, len(time), 10)
plt.plot(np.arange(0, Nx, dx), T[0], '-k', label = "t = {:.2f} s".format(0))
for i in range(len(ind)-1):
    t=ind[i]*dt
    plt.plot(np.arange(0, Nx, dx), T[int(ind[i])], '-k', alpha=0.2, label = "t = {:.2f} s".format(t))
    #plt.legend()

plt.plot(np.arange(0, Nx, dx), T[-1], '-r', label = "t = {:.2f} s".format(t_final))
plt.xlabel("Length [m]")
plt.ylabel('Temperature [°C]')
plt.title(r'Temperature as function of time', fontsize = 15)
plt.legend(fontsize=7)
plt.show()
plt.savefig('alu_1D.png')

'''
fig = plt.figure()
ax = fig.add_axes([0.12, 0.12, 0.80, 0.80])
line1, = plt.plot([], [], '-k', alpha=0.3)
line2, = plt.plot([], [], '-k', alpha=1)
plt.xlim(0, Nx)
plt.ylim(-100, 100)
plt.xlabel("Length [m]")
plt.ylabel('Temperature [°C]')
title = plt.text(0.60, 0.84, r'Temperature as function of time', transform=ax.transAxes, ha='center')

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    return line1,line2

def animate(i) :
    line1.set_data(np.arange(0, Nx, dx), T[i])
    line2.set_data(np.arange(0, Nx, dx), T[0])
    title.set_text("                                                                     t = {:.2f} s".format(time[i]))
    return line1, line2, title

inter = 3
ani = FuncAnimation(fig, animate, init_func=init, frames=T.shape[0], blit=True, interval=inter, repeat=True)
plt.show()
'''
#writergif = PillowWriter(fps=25)
#ani.save('1D_alu.gif',writer=writergif)
