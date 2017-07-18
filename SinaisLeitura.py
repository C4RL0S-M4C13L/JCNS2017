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
# SinaisLeitura.py : Functions and tools to read data files. Labels
# are provided on the file "dados.descricao"

from numpy import loadtxt,shape,histogram,digitize
from scipy.signal import detrend

def Canais(Nome):
	descricao = open('dados.descricao', 'r')
	fs = []
	canais = []
	print(Nome)
	for line in descricao:
		F  = line.split()
		ruim = True
		if 'BAD' not in F:
			if Nome == F[0]:
				canais = (F[3], F[4], F[5], F[6])
				print('Channels: ', canais)
				fs = float(F[2])*1000
				print( 'F Sampl : ', fs)
				ruim = False
				break
	descricao.close()
	if ruim == True:
		fs = []
		canais = []
		print( ' Data does not exist !!!')
	return fs, canais, ruim

def CanaisBaseEstAn(Nome):
	descricao = open('dados.descricao', 'r')
	fs = []
	canais = []
	for line in descricao:
		F  = line.split()
		ruim = True
		if 'BAD' not in F:
			if Nome == F[0]:
				canais = (F[3], F[4], F[5], F[6])
				fs = float(F[2])*1000
				ruim = False
				break
	descricao.close()
	if ruim == True:
		fs = []
		canais = []
	return F[1],F[7]
	
def sinais(Nome, inicio=0, fim=0): 
	Xi  = loadtxt('./dados/'+Nome+'.txt.gz')
	[a,b] = shape(Xi)
	if b == 4:
		Xi = Xi.T
	if inicio == 0:
		inicio = 1
	if fim == 0:
		fim = len(Xi.T)
	Xi = Xi[:,inicio:fim]
	for i in range(4):
		Xi[i]=detrend(Xi[i])
	return Xi

def segmentacao(Nome, Fs, janela):
	X  = loadtxt(Nome+'.txt.gz')
	[a,b] = shape(X)
	if b == 4:
		X = X.T
	ch1 = X[0,:]
	ch2 = X[1,:]
	ch3 = X[2,:]
	ch4 = X[3,:]
	t = arange(len(ch1))/Fs
	if '200' in Canais:
		x = X[Canais.index('200'),:]
	elif '27' in Canais:
		x = X[Canais.index('27'),:] 
	elif '58' in Canais:
		x = X[Canais.index('58'),:]
	xmax = max(abs(x))
	t0 = 0
	while abs(x[t0])<xmax/3:
		t0=t0+1	
	t0i = t0
	lista = []
	nj = round(log2(janela*Fs/1000))
	njanela = int(2**nj)
	lista = [t0, t0+njanela] 
	eh_estacionario = False
	while not eh_estacionario and t0 < len(x) / 2:
		eh_estacionario = Medidas().eh_estacionarios(ch1[t0:t0+njanela], alfa) \
		              and Medidas().eh_estacionarios(ch2[t0:t0+njanela], alfa) \
		              and Medidas().eh_estacionarios(ch3[t0:t0+njanela], alfa) \
		              and Medidas().eh_estacionarios(ch4[t0:t0+njanela], alfa)
		t0 = t0 + 100
		lista = [t0, t0+njanela] 
	return lista, t0i, njanela/Fs*1000

def discretizar(X,NBins):
	for i in range(4):
		h,edges = histogram(X[i], NBins)
		X[i]=digitize(X[i], edges)-1
	return X
