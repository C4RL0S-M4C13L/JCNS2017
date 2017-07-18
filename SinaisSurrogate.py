#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2008-2009 Signal Processing Laboratory - São Carlos School of Engineering
# Authors: Fernando Pasquini Santos (fernando.pasquini@ufu.br)
#          Carlos Dias Maciel (cdmaciel@usp.br)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is experimental, and distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# SinaisLeitura.py : Functions and tools to generate surrogate
# data given a time-series

from numpy import fft, angle, random, exp, array, sort, zeros, sum, savetxt, loadtxt
from time import clock
from sys import argv
import pyfftw

def erro(X,Y):
	n = len(X)
	X2 = array(X)
	Y2 = array(Y)
	return sum((X2-Y2)*(X2-Y2))/sum(Y2*Y2)	

def substitui(x,y):
	sy  = sort(y)
	sx  = sort(x)
	dic = {}
	for i in range(len(x)):
		if sy[i] not in dic:
			dic[sy[i]] = sx[i]
	saida = zeros(len(x))
	for i in range(len(x)):
		saida[i] = dic[y[i]]
	return saida

def AAFT(y, Tol):
	MaxIter = 50
	Y = fft.fft(y)
	absY  = abs(Y)           #numpy.fft.fft(y))
	angY  = angle(Y)   #numpy.fft.fft(y))
	ry    = random.permutation(y)
	RY    = fft.fft(ry)
	absRY = abs(RY)
	j = complex(0,1)
	i = 0
	Erro = []
	ok = False
	start= clock()
	time = 0
	while ok == False:   #erro(absRY,absY) > 0.001:
		angRY = angle(RY)
		Z     = absY*exp(j*angRY)
		mry   = fft.ifft(Z)
		ry    = substitui(y, mry)
		RY    = fft.fft(ry)
		absRY = abs(RY)
		Erro.append(erro(absRY,absY))
		if i == 0:
			MinErro = Erro[i]
		if Erro[i] < MinErro:
			saida = mry.real # [float(i) for i in mry.real()]
			MinErro = Erro[i]
			end= clock()
			time= end-start
		if i>0:
			if abs(Erro[i]-Erro[i-1]) <Tol:
				ok = True
		if i > MaxIter:
			ok = True
		i = i + 1
	st =  'Converged in '+str(i)+' iteractions; rms error ' + "{0:.5f}".format(Erro[i-1]) + ' in ' + "{0:.2f}".format(time) + ' s'
	return saida, st

def PhaseRandom(y):
	N = len(y)
	y = y - scipy.stats.mean(y)
	Y = scipy.fftpack.basic.fft(y)
	absY  = abs(Y)
	i = complex(0,1)
	theta = numpy.zeros(N,'float')
	if N % 2 == 1:
		theta[N//2+1] = 0
		for j in range(1,N//2):
			theta[j] = numpy.random.random()*numpy.pi
			theta[N - j] = -1*theta[j]
	else:
		for j in range(1,N//2):
			theta[j] = numpy.random.random()*numpy.pi
			theta[N - j] = -1*theta[j]
		YR = absY*(10**(theta*i))
	y1 = scipy.fftpack.basic.ifft(YR)
	saida  = numpy.real_if_close(y1,100)
	return saida
	
def AAFT_fftw(y, Tol):
	MaxIter = 50
	Y = pyfftw.interfaces.numpy_fft.fft(y)
	absY  = abs(Y)           #numpy.fft.fft(y))
	angY  = angle(Y)   #numpy.fft.fft(y))
	ry    = random.permutation(y)
	RY    = pyfftw.interfaces.numpy_fft.fft(ry)
	absRY = abs(RY)
	j = complex(0,1)
	i = 0
	Erro = []
	ok = False
	start= clock()
	time = 0
	while ok == False:   #erro(absRY,absY) > 0.001:
		angRY = angle(RY)
		Z     = absY*exp(j*angRY)
		mry   = pyfftw.interfaces.numpy_fft.ifft(Z)
		ry    = substitui(y, mry)
		RY    = pyfftw.interfaces.numpy_fft.fft(ry)
		absRY = abs(RY)
		Erro.append(erro(absRY,absY))
		if i == 0:
			MinErro = Erro[i]
		if Erro[i] < MinErro:
			saida = mry.real # [float(i) for i in mry.real()]
			MinErro = Erro[i]
			end= clock()
			time= end-start
		if i>0:
			if abs(Erro[i]-Erro[i-1]) <Tol:
				ok = True
		if i > MaxIter:
			ok = True
		i = i + 1
	st =  'Converged in '+str(i)+' iteractions; rms error ' + "{0:.5f}".format(Erro[i-1]) + ' in ' + "{0:.2f}".format(time) + ' s'
	return saida, st

if __name__ == '__main__':
	name = argv[1] #'ID-XXXX'
	ch = int(argv[2])
	num = int(argv[3])
	X = loadtxt('dados/'+name+'.txt.gz')
	xsur,msg = AAFT_fftw(X[ch,:],0.001)
	savetxt('surrogates/'+name+'_sur_%d.txt.gz'%num,xsur)