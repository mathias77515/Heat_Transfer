import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter




class Tools_matrix1D(object):

    def __init__(self, T_left, T_right):
        self.Tleft = T_left
        self.Tright = T_right

    def initialize_bar(self, N, dx, T):

        var = np.linspace(0, 2*np.pi, N)
        X = T * var**2
        return X

    def initialize_temp(self, X, left, right, init):

        if init:
            X[:] = self.Tinit
        if left:
            X[0] = self.Tleft
        if right:
            X[-1] = self.Tright
        return X


    def Hx_matrix1D(self, N, d):
        H = np.zeros((N,N))

        for i in range(N):
            H[i,i]=2/d**2
            if i!=N-1:
                H[i,i+1] = -1/d**2
            if i!=0:
                H[i,i-1] = -1/d**2
        return H

    def border(self, X):
        X[0] = self.Tleft
        X[-1] = self.Tright
        return X

    def update(self, X, Mat, dt, kappa):
        return X - dt*kappa*Mat@X

class Plots:
    def __init__(self, dt, Nx):
        self.dt = dt
        self.Nx = Nx

    def Plot1D(self, var, X, time, name):
        plt.figure()
        t=0
        ind = np.linspace(0, len(time), 10)
        plt.plot(var, X[0], '-k', label = "t = {:.2f} s".format(0))
        for i in range(len(ind)-1):
            t=ind[i]*self.dt
            plt.plot(var, X[int(ind[i])], '-k', alpha=0.2, label = "t = {:.2f} s".format(t))

        plt.plot(var, X[-1], '-r', label = "t = {:.2f} s".format(time[-1]))
        plt.xlabel("Length [m]")
        plt.ylabel('Temperature [°C]')
        plt.title(r'Temperature as function of time', fontsize = 15)
        plt.legend(fontsize=7)
        if type(name)==str:
            plt.savefig('{}.png'.format(name))
        plt.show()

    def Plot2D(self, X, name):
        plt.figure()
        plt.imshow(X[-1], cmap = 'bwr')
        plt.xlabel(r"$N \times dx$ [m]")
        plt.ylabel(r'$N \times dy$ [m]')
        if type(name)==str:
            plt.savefig('{}.png'.format(name))
        plt.show()

    def anim1D(self, var, X, time, name):

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.12, 0.80, 0.80])
        line1, = plt.plot([], [], '-k', alpha=0.3)
        line2, = plt.plot([], [], '-k', alpha=1)
        plt.xlim(0, self.Nx)
        plt.ylim(np.min(X)-10, np.max(X)+10)
        plt.xlabel("Length [m]")
        plt.ylabel('Temperature [°C]')
        title = plt.text(0.60, 0.84, r'Temperature as function of time', transform=ax.transAxes, ha='center')

        def init():
            line1.set_data([],[])
            line2.set_data([],[])
            return line1,line2

        def animate(i) :
            line1.set_data(var, X[i])
            line2.set_data(var, X[0])
            title.set_text("                                                                     t = {:.2f} s".format(time[i]))
            return line1, line2, title

        inter = 1
        ani = FuncAnimation(fig, animate, init_func=init, frames=X.shape[0], blit=True, interval=inter, repeat=True)
        plt.show()

        if type(name)==str:
            writergif = PillowWriter(fps=25)
            ani.save('{}.gif'.format(name),writer=writergif)


class Tools_matrix2D(object):

    def __init__(self, T_init, T_up, T_down, kappa, dx, dy, dt, l):
        self.Tinit = T_init
        self.Tup = T_up
        self.Tdown = T_down
        self.kappa = kappa
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.lamb = l

    def Hx_matrix2D(self, N, d):
        H = np.zeros((N**2, N**2))
        for i in range(N**2):
            H[i,i] = 2/d**2
        for i in range(1, (N**2)-1):
            H[i,i-1] = -1/d**2
            H[i,i+1] = -1/d**2
        for i in range(N**2):
            if i%N == 0:
                H[i,i-1] = 0/d**2
                H[i-1,i] = 0/d**2

        H[0,1] = -1/d**2
        H[-1, -2] = -1/d**2
        return H

    def initialize_temp(self, X, up, down, init):

        if init:
            X[1:-1,:] = self.Tinit
        if up:
            X[0,:] = self.Tup
        if down:
            X[-1,:] = self.Tdown
        return X

    def update2D(self, T, i, Hx, Hy):
        X = np.reshape(T[i], (T[0].shape[0]*T[0].shape[1]))
        Y = np.reshape(T[i], (T[0].shape[0]*T[0].shape[1]), order = 'F')
        termx = Hx@X
        termy = Hy@Y
        termx = np.reshape(termx, T[i].shape)
        termy = np.reshape(termy, T[i].shape, order = 'F')
        T[i, 0] = self.Tup
        T[i, -1] = self.Tdown
        return T[i] - self.dt*self.kappa*(termx+termy)

    def flux(self, N, T, S, dx, dy):

        MatGradx = np.zeros((N**2, N**2))
        MatGrady = np.zeros((N**2, N**2))

        for i in range(0, (N**2)-1):
            MatGradx[i,i]= -1/dx
            MatGradx[i,i+1]= 1/dx
            MatGrady[i,i]= -1/dy
            MatGrady[i+1,i]= 1/dy

        X = np.reshape(T, (T.shape[0]*T.shape[1]))
        Y = np.reshape(T, (T.shape[0]*T.shape[1]), order = 'F')

        gradTx = np.reshape(MatGradx @ X, T.shape)
        gradTy = np.reshape(MatGrady @ Y, T.shape, order = 'F')

        gradTx[:, 0:1] = 0
        gradTy[:, 0:1] = 0
        gradTx[:, -1:-2] = 0
        gradTy[:, -1:-2] = 0

        return -self.lamb * gradTx/S, -self.lamb * gradTy/S





    def print_constante_2D(self, array, time):
        print("========== Constantes ===========\n")
        print("   kappa = {:.10f} m2.s-1".format(self.kappa))
        print("   dx = {} ; dy = {}".format(self.dx, self.dy))
        print("   dt = {}".format(self.dt))
        print("   Time fo the simulation : {} s".format(time))
        print("   Size of temperature array : ({}, {})".format(array.shape[0], array.shape[1]))
        print("\n=================================\n")


class Tools(object):

    def __init__(self, T_init, T_down, T_up, T_side, kappa, dt, dx, dy, mu, rho, g, beta):
        self.Tinit = T_init
        self.Tdown = T_down
        self.Tup = T_up
        self.Tside = T_side
        self.kappa = kappa
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.mu = mu
        self.rho = rho
        self.g = g
        self.beta = beta

    def initialize_temp(self, array):
        try:
            array.shape[1]
        except IndexError:
            print("Give a 2D array !")

        array[0] = self.Tup
        array[-1] = self.Tdown
        #array[1:-1, :] = self.Tside

        return array

    def update_temp(self, time, array, vx, vy, indx, indy, adv):

        temp_x = 0
        temp_y = 0

        if indx == 0:
            temp_x = ((self.kappa)/(self.dx**2)) * (array[indx+1,indy]-array[indx,indy])

        if indy == 0:
            temp_y = ((self.kappa)/(self.dy**2)) * (array[indx,indy+1]-array[indx,indy])

        if indx == time:
            temp_x = ((self.kappa)/(self.dx**2)) * (array[indx-1,indy]-array[indx,indy])

        if indy == time:
            temp_y = ((self.kappa)/(self.dy**2)) * (array[indx,indy-1]-array[indx,indy])

        else:
            temp_x = ((self.kappa)/(self.dx**2)) * (array[indx+1,indy]+array[indx-1,indy]-2*array[indx,indy])
            temp_y = ((self.kappa)/(self.dy**2)) * (array[indx,indy+1]+array[indx,indy-1]-2*array[indx,indy])



        if adv:
            advx = vx[indx,indy] * (array[indx+1,indy] - array[indx,indy])/self.dx
            advy = vy[indx,indy] * (array[indx,indy+1] - array[indx,indy])/self.dy
        else:
            advx = 0
            advy = 0


        return array[indx,indy]+self.dt*(temp_x+temp_y-advx-advy)

    def update_vx(self, vx, vy, i, j):
        return vx[i,j] - ((vy[i,j+1] - vy[i,j])*self.dx/self.dy)

    def update_vy(self, array, vx, vy, i, j):
        betag = self.beta*self.g
        murho = self.mu*self.rho
        term1 = betag * (array[i,j] - self.Tdown)
        term2 = murho*((vy[i,j-1]+vy[i,j+1]-2*vy[i, j])/self.dy**2 + (vx[i-1,j]+vx[i+1,j]-2*vx[i, j])/self.dx**2)
        term3 = vx[i,j]*(vy[i,j+1]-vy[i,j])/self.dx
        term4 = vy[i,j]*(vy[i,j+1]-vy[i,j])/self.dy
        #print(term1)
        #print(term2)
        #print(term3)
        #print(term4)
        #print()
        return vy[i, j] + self.dt * (term1 + term2 - term3 - term4)

    def print_constante(self, array, time):
        print("========== Constantes ===========\n")
        print("   kappa = {:.10f} m2.s-1".format(self.kappa))
        print("   dx = {} ; dy = {}".format(self.dx, self.dy))
        print("   dt = {}".format(self.dt))
        print("   Time fo the simulation : {} s".format(time))
        print("   Size of temperature array : ({})".format(len(array)))
        print("\n=================================\n")
