# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:50:37 2020

reference: https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot

@author: bruce
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy import sin, cos
bones = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
         (22, 23), (23, 8), (24, 25), (25, 12))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 5)
'''
atext = fig.add_subplot(111, projection='3d')
# make the panes transparent
atext.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
atext.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
atext.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
atext.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
atext.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
atext.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
'''
skl_file = 'VS001C001P001R002A030'
xs, ys, zs = np.loadtxt(skl_file+'.txt', delimiter=' ', usecols=(0,1,2), unpack=True)
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
theta = np.radians(0)
rotation_y_matrix = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
rotation_y_matrix_np = np.array(rotation_y_matrix)
length=136
scale=1.1
#ax.scatter((zs[0]), (xs[0]), (ys[0]),  color='g')
xs=-xs  # horizontal flip
for i in range(len(xs)):
    xyz_rot=rotation_y_matrix_np.dot(np.array([xs[i],ys[i],zs[i]]))
    xs[i] = xyz_rot[0]
    ys[i] = xyz_rot[1]
    zs[i] = xyz_rot[2]
    
for p in range(0,length):
    move=[xs[0+p*25] - xs[0],ys[0+p*25] - ys[0],zs[0+p*25] - zs[0]]
    for i in range(0,25): # plot each point + it's index as text above
        x = xs[i+p*25] - xs[0]
        y = ys[i+p*25] - ys[0]
        z = zs[i+p*25] - zs[0]
        #xyz_rot=rotation_y_matrix_np.dot(np.array([[x],[y],[z]]))
        
        #xyz_rot_scale = 1.1*xyz_rot
        label = i+1
        if p in [0]:
            ax.scatter(z, x, y, color='#40ff00')
        else:
            ax.scatter(z, x, y, color=str((length-p)/length))
        if p > 0:
            ax.plot([z,zs[i+(p-1)*25] - zs[0]],[x,xs[i+(p-1)*25] - xs[0]],[y,ys[i+(p-1)*25] - ys[0]],color=str((length-p)/length))
        #ax.scatter(z*2, x*2, y*2, color='g')
        #ax.scatter(z*scale-move[2]*(scale-1), x*scale-move[0]*(scale-1), y*scale-move[1]*(scale-1), color='b')
        #ax.scatter(z*2,x*2, y*2,  color='pink')
        #ax.scatter(xyz_rot[2], xyz_rot[0], xyz_rot[1],  color=str((length-p)/length))
        #ax.scatter(xyz_rot_scale[2], xyz_rot_scale[0], xyz_rot_scale[1],  color='black')
        #ax.text(z, x, y, '%s' % (label), size=10, zorder=2, color='b')
for p in [0]:
    for (b1,b2) in bones:
        b1=b1-1
        b2=b2-1
        x1 = xs[b1+p*25] - xs[0]
        y1 = ys[b1+p*25] - ys[0]
        z1 = zs[b1+p*25] - zs[0]
        x2 = xs[b2+p*25] - xs[0]
        y2 = ys[b2+p*25] - ys[0]
        z2 = zs[b2+p*25] - zs[0]
        ax.plot([z1,z2],[x1,x2],[y1,y2],color = '#40ff00')
    
'''
x0=xs[25]-xs[0]
y0=ys[25]-ys[0]
z0=zs[25]-zs[0]
scale=2
for i in range(25,40):
  x = xs[i] - xs[0]
  y = ys[i] - ys[0]
  z = zs[i] - zs[0]
  #xyz_rot=rotation_y_matrix_np.dot(np.array([[x],[y],[z]]))
  xyz_rot=rotation_y_matrix_np.dot(np.array([x,y,z]))
  xyz_rot_scale = 1.1*xyz_rot
  label = i+1
  ax.scatter(z, x, y,  color='g')
  #ax.scatter(z*1.2, x*1.2, y*1.2,  color='b')
  
  ax.scatter((z-z0)*scale+z0, (x-x0)*scale+x0, (y-y0)*scale+y0,  color='b')
  #ax.scatter((z-zs[25])*1.2+zs[25], (x-xs[25])*1.2+xs[25], (y-ys[25])*1.2+ys[25],  color='b')
  #ax.scatter(z*0.9, x*0.9, y*0.9,  color='pink')
  #ax.scatter(xyz_rot[2], xyz_rot[0], xyz_rot[1],  color='r')
  #ax.scatter(xyz_rot_scale[2], xyz_rot_scale[0], xyz_rot_scale[1],  color='black')
  #ax.text(z, x, y, '%s' % (label), size=10, zorder=2, color='b')
ax.scatter((zs[25]), (xs[25]), (ys[25]),  color='r') 
ax.text((zs[25]), (xs[25]), (ys[25]), '%s' % (25), size=10, zorder=2, color='r')
ax.scatter((zs[0]), (xs[0]), (ys[0]),  color='r')
ax.scatter(0, 0, 0,  color='black')
'''

#ax.text(0, 0, 0, '%s' % ('O'), size=12, zorder=90000, color='r')
#ax.scatter(0, 0, 0,  color='r')
#ax.scatter(zs, xs, ys, marker='o')
ax.set_xlabel('Z')#, fontsize=10
ax.tick_params(axis='z', labelsize=7)
ax.set_ylabel('X')
ax.tick_params(axis='x', labelsize=7)
ax.set_zlabel('Y')
ax.tick_params(axis='y', labelsize=7)
plt.draw()
plt.savefig(skl_file+'.png', dpi=300)

'''
for angle in range(0, 360):
  ax.view_init(30, angle)
  plt.draw()
  plt.pause(1)

m = rand(3,3) # m is an array of (x,y,z) coordinate triplets

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(m)): # plot each point + it's index as text above
  x = m[i,0]
  y = m[i,1]
  z = m[i,2]
  label = i
  ax.scatter(x, y, z, color='b')
  ax.text(x, y, z, '%s' % (label), size=25, zorder=1, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for angle in range(0, 360):
  ax.view_init(30, angle)
  plt.draw()
  plt.pause(.001)
'''