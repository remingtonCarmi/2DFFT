#!/usr/local/bin/python
from math import *
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import errno
import numpy as np

## 2DFFT.py
## Remi Carmigniani
#
directory = 'result'
if not os.path.exists(directory):
    os.makedirs(directory)


##################################################################################################################
############################			Useful functions		      ############################
##################################################################################################################
#Central differentiation 2nd order
def cScheme(u,dx):
	nx = len(u)
	ny = len(u[1])
	result = [[0 for i in range(ny)] for j in range(nx)]
	for i in range(nx-1):
		for j in range(ny):
			result[i][j] =(u[i+1][j]-u[i-1][j])/(2*dx)
	for j in range(ny):	
		result[0][j]=(u[1][j]-u[nx-1][j] )/(2*dx)
		result[nx-1][j]=(u[0][j]-u[nx-2][j] )/(2*dx)
	return result
def cSchemey(u,dx):
	nx = len(u)
	ny = len(u[1])
	result = [[0 for i in range(ny)] for j in range(nx)]
	for i in range(nx):
		for j in range(ny-1):
			result[i][j] =(u[i][j+1]-u[i][j-1])/(2*dx)
	for i in range(ny):	
		result[i][0]=(u[i][1]-u[i][ny-1] )/(2*dx)
		result[i][ny-1]=(u[i][0]-u[i][ny-2] )/(2*dx)
	return result
#L2 error using Simpson rule 
def errorL2(f,dx,dy,nx,ny):
	error = f[0][0]+f[0][ny-1]+f[nx-1][0]+f[nx-1][ny-1]
	for j in range(1,int((ny-1)/2)+1):
		error = error+ 4.*(f[0][2*j-1]+f[nx-1][2*j-1])
	for j in range(1,int((ny-1)/2)):
		error = error+ 2.*(f[0][2*j]+f[nx-1][2*j])
	for i in range(1,int((nx-1)/2)+1):
		error = error+ 4.*(f[2*i-1][0]+f[2*i-1][ny-1])
	for i in range(1,int((nx-1)/2)):
		error = error+ 2.*(f[2*i][0]+f[2*i][ny-1])
	for i in range(1,int((nx-1)/2)+1):
		for j in range(1,int((ny-1)/2)+1):
			error = error + 16.*f[2*i-1][2*j-1]
        for i in range(1,int((nx-1)/2)+1):
		for j in range(1,int((ny-1)/2)):
			error = error + 8.*f[2*i-1][2*j]
        for i in range(1,int((nx-1)/2)):
		for j in range(1,int((ny-1)/2)+1):
			error = error + 8.*f[2*i][2*j-1]
        for i in range(1,int((nx-1)/2)):
		for j in range(1,int((ny-1)/2)):
			error = error + 4.*f[2*i][2*j]		
	return (error*1./9.*dx*dy)


#plot figure
def plotFig(u_arr,numb,t,x,y):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X, Y = np.meshgrid(x, y)
	zmin = min(min(u_arr))
	zmax = max(max(u_arr))
	v = np.linspace(-1.2, 1.2, 15, endpoint=True)
	surf = ax.plot_surface(X, Y, u_arr,v, rstride=1, cstride=1,cmap=cm.jet,
 	       linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
	ax.set_zlim(zmin, zmax)
	ax.set_xlim(0, L)
	ax.set_ylim(0, L)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf)
	fig.suptitle('t = '+ repr(t)) 
	plt.savefig(directory +'/t'+'%0*d' % (3, numb)+'.png')



## Test function
def test(x,y):
    return 1./(3.+0.5*(cos(x)+0.5*sin(4.*y))*(sin(x)+cos(6.*y)))
## Chech derivatives
def testp(x,y):
	return -((-0.5*sin(x)*(cos(6.*y) + sin(x)) + 0.5 *cos(x)*(cos(x) + 0.5 *sin(4.*y)))/(3 +0.5*(cos(6.*y) + sin(x))*(cos(x) + 0.5 *sin(4.*y)))**2)
## Chech derivatives
def testpy(x,y):
	return -((cos(4.*y)*(cos(6.*y) + sin(x)) -3.*(cos(x) + 0.5*sin(4.*y))*sin(6.*y))/(3 + 0.5*(cos(6.*y) + sin(x))*(cos(x) + 0.5*sin(4.*y)))**2)
#Trap Integration (Copied from 1DDiff.py)
def trapInt(f,dx,nx):
	error = 0.0
	for i in range(0,nx-1):
		error = error+f[i]+f[i+1]	
	return error*.5*dx

def realPart(x,nx,ny):
	return [[x[j][i].real for i in range(ny)] for j in range(nx)]

def FFT(u):
	return np.fft.fft2(u) 
def InvFFT(utild):
	#n=len(utild)-1
	#return [sum([1./float(n+1)*utild[i]*cmath.exp(1j*2*pi*i*k/float(n+1)) for i in range(n/2)])+sum([1./float(n+1)*utild[i]*cmath.exp(1j*2*pi*(i-n-1)*k/float(n+1)) for i in range(n/2,n+1)]) for k in range(n+1)]
	return np.fft.ifft2(utild)

def power(a,b):
	if abs(b*b)>0:
		if abs(a*a)>0:
			return cmath.exp(b*cmath.log(a))
		else:
			return 0
	else: 
		return 1;
def DFFT(utild,orderx,ordery): 
	nx = len(utild)
	ny = len(utild[1])
	result = [[0 for i in range(ny)] for j in range(nx)]
	for i in range(nx/2):
		for j in range(ny/2):
			result[i][j] = utild[i][j]*power(1j*i,orderx)*power(1j*j,ordery)
		for j in range(ny/2,ny):
			result[i][j] = utild[i][j]*power(1j*i,orderx)*power(1j*(j-ny),ordery)
	for i in range(nx/2,nx):
		for j in range(ny/2):
			result[i][j] = utild[i][j]*power(1j*(i-nx),orderx)*power(1j*j,ordery)
		for j in range(ny/2,ny):
			result[i][j] =utild[i][j]*power(1j*(i-nx),orderx)*power(1j*(j-ny),ordery)
	return result
		
	
##################################################################################################################
############################				End			      ############################
##################################################################################################################


## Discretization parameter 
N = [9,15,33,51,101,201]
L=2*pi
error=[]
errory=[]
errorC =[]
errorCy =[]
for k in range(0,6):
	Nx=N[k]
	Ny=N[k]
	dx=L/float(Nx)
	dy=L/float(Ny)
	x = [dx*ii for ii in range(Nx)] 
	y = [dy*ii for ii in range(Ny)] 
	#Construct the solution
	u = [[test(x[jj],y[ii]) for ii in range(Nx)] for jj in range(Ny)]
	up = [[testp(x[jj],y[ii]) for ii in range(Nx)] for jj in range(Ny)]
	#plotFig(up,k,k,x,y)
	#Calculate the FFT
	utild  = FFT(u)
	#Calculate the derivative
	upxtild = DFFT(utild,1,0) #order in the two directions 1,0 means here df/dx
	upxtild2= FFT(up)
	#Calculate the InverseFFT
	ur = realPart(InvFFT(upxtild),Nx,Ny) 
	#plotFig(ur,k+10,k+10,x,y)
	#Calculate the error
	err = cmath.sqrt(errorL2([[(testp(x[ii],y[jj])-ur[ii][jj])**2  for ii in range(Nx)] for jj in range(Ny)],dx,dy,Nx,Ny)).real
	error.append(err)
	print 'Calculated the error  for dw/dx ' + repr(Nx) + ' with FFT : ' + repr(err)
        #Calculate the derivative
	upytild = DFFT(utild,0,1) #order in the two directions 0,1 means here df/dy
	ury = realPart(InvFFT(upytild),Nx,Ny) 
        err = cmath.sqrt(errorL2([[(testpy(x[ii],y[jj])-ury[ii][jj])**2  for ii in range(Nx)] for jj in range(Ny)],dx,dy,Nx,Ny)).real
	errory.append(err)
	print 'Calculated the error  for dw/dy ' + repr(Nx) + ' with FFT : ' + repr(err)
	upc = cScheme(u,dx)
	err = cmath.sqrt(errorL2([[(testp(x[ii],y[jj])-upc[ii][jj])**2  for ii in range(Nx)] for jj in range(Ny)],dx,dy,Nx,Ny)).real
 	errorC.append(err)
	print 'Calculated the error  for dw/dx ' + repr(Nx) + ' with central Scheme : ' + repr(err)
        upcy= cSchemey(u,dy)
	err = cmath.sqrt(errorL2([[(testpy(x[ii],y[jj])-upcy[ii][jj])**2  for ii in range(Nx)] for jj in range(Ny)],dx,dy,Nx,Ny)).real
 	errorCy.append(err)
	print 'Calculated the error  for dw/dy ' + repr(Nx) + ' with central Scheme : ' + repr(err)


plt.loglog(N, error,label='FFT dx')
plt.loglog(N, errory,label='FFT dy')
plt.loglog(N, errorC,label='Central dx')
plt.loglog(N, errorCy,label='Central dy')
plt.legend(loc=3)
plt.title('Error')
plt.savefig('Error.png')

print 'Simulation Completed without error'


