import numpy as np

x = np.linspace(-2,2,200)
y = np.linspace(-1,3,200)

X,Y = np.meshgrid(x,y)

Z = (1-X)**2 + 100*(Y-X**2)**2


sgd_traj = run_optimizer(SGD(lr=0.001))
adam_traj = run_optimizer(Adam(lr=0.05))
adamw_traj = run_optimizer(AdamW(lr=0.05))



import matplotlib.pyplot as plt

plt.contour(X,Y,Z,levels=50)

sgd_traj = np.array(sgd_traj)
adam_traj = np.array(adam_traj)
adamw_traj = np.array(adamw_traj)

plt.plot(sgd_traj[:,0],sgd_traj[:,1],label="SGD")
plt.plot(adam_traj[:,0],adam_traj[:,1],label="Adam")
plt.plot(adamw_traj[:,0],adamw_traj[:,1],label="AdamW")

plt.legend()
plt.title("Optimizer trajectories on Rosenbrock function")

plt.show()
