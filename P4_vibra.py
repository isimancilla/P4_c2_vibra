import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from mpmath import besseljzero, besselj

t_values = np.linspace(0,1, 50)
alpha = np.pi / 4
r_values = np.linspace(0, 1, 50)
theta_values = np.linspace(0, 2 * np.pi - alpha, 50)

A = 1
R = 1
m1 = np.pi / (2* np.pi - alpha) # con n = 1
m2 = 2 * np.pi / (2* np.pi - alpha) #con n = 2
C = 10

R_mesh, Theta_mesh = np.meshgrid(r_values,theta_values)
X, Y = R_mesh*np.cos(Theta_mesh), R_mesh*np.sin(Theta_mesh)

x, y = r_values*np.cos(theta_values), R_mesh*np.sin(theta_values)


bessel = np.frompyfunc(lambda *a: float(besselj(*a)), 2, 1)
ceros_bessel = np.frompyfunc(lambda *a: float(besseljzero(*a)), 2, 1)

def f1(r,theta,t, k, m):
    cero = ceros_bessel(m,k)
    w = C / R * cero
    output = A * bessel(m, w * r/ C) * np.sin(theta*m) * np.sin(w * t)
    return output.astype(np.float)

# Modo con m = 0 y k = 1

fig = plt.figure(1)
ax = fig.gca(projection='3d')
def animacion(i):
    ax.clear()
    Z = f1(R_mesh, Theta_mesh,t_values[i], 1, m1)
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_title('Modo normal con m = 1, k = 1')
    ax.set_zlim(-1,1)
ani = animation.FuncAnimation(fig, animacion, range(len(t_values)), interval=1, repeat=False)

plt.show()

writer_gif= animation.PillowWriter(fps=60)
ani.save('gif_1.gif', writer=writer_gif)

#modo con m = 2 y k = 2

fig2 = plt.figure(2)
ax2 = fig.gca(projection='3d')
def animacion2(i):
    ax2.clear()
    Z2 = f1(R_mesh, Theta_mesh,t_values[i], 2, m2)
    ax2.plot_surface(X, Y, Z2, cmap='plasma')
    ax2.set_title('Modo normal con m = 2, k = 2')
    ax2.set_zlim(-1,1)
ani2 = animation.FuncAnimation(fig, animacion2, range(len(t_values)), interval=1, repeat=False)

plt.show()

ani2.save('gif_2.gif', writer=writer_gif)