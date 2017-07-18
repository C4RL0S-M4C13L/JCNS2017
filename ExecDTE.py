#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2008-2017 Signal Processing Laboratory - São Carlos School of Engineering
# Authors: Fernando Pasquini Santos (fernando.pasquini@ufu.br)
#          Carlos Dias Maciel (carlos.maciel@usp.br)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is experimental, and distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# ExecDTE.py : Executes delayed transfer entropy measures given an
# input file called "dados.execucao". Data files have to be on
# a folder called "dados" (see SinaisLeitura.py)

from itertools import combinations, permutations
import time
import pickle 
from datetime import datetime
from scipy import signal
import scipy.stats as ss
import os.path
import numpy as np

import SinaisMedidas
import SinaisLeitura
import SinaisSurrogate

from ipyparallel import Client
rc = Client()		
mec = rc[:] 
mec.block=True
with  mec.sync_imports():
	from numpy import loadtxt, arange, fft, angle, random, exp, array, sort, zeros, sum, histogramdd, log2, split, sqrt, shape, unique, floor, ceil, vstack, roll
	from itertools import product
	from scipy import signal

def TExy_norm(xd,yds,y,valor,NBins): #Transferencia de entropia de x para y
	"""
	"""
	pxdydsy, edges = histogramdd(vstack((xd,yds,y)).T, bins = NBins, normed=True,range = [valor[0]]+[valor[1]]*(len(yds)+1))
	pxdydsy = pxdydsy/sum(sum(sum(pxdydsy)))
	pxdyds, edges   = histogramdd(vstack((xd,yds)).T, bins = NBins, normed=True, range = [valor[0]]+[valor[1]]*len(yds))
	pxdyds = pxdyds/sum(pxdyds)
	pydsy, edges   = histogramdd(vstack((yds,y)).T, bins = NBins, normed=True, range = [valor[1]]*len(yds)+[valor[1]])
	pydsy = pydsy/sum(pydsy)
	if len(shape(yds))>1:
		pyds, edges   = histogramdd(yds.T, bins = NBins, normed=True, range = [valor[1]]*len(yds))
	else:
		pyds, edges   = histogramdd([yds.T], bins = NBins, normed=True, range = [valor[1]])
	pyds = pyds/sum(pyds)

	TExy = 0
	escala = range(NBins)
	
	# i - [xd, yds, y]
	for i in product(range(NBins),repeat=len(yds)+2):
		if pxdydsy[i]!=0 and pxdyds[i[:-1]]!=0 and pydsy[i[1:]]!=0 and pyds[i[1:-1]]!=0:
			TExy = TExy + pxdydsy[i]*(log2(pxdydsy[i]*pyds[i[1:-1]]/(pydsy[i[1:]]*pxdyds[i[:-1]])))
	
	# normalization
	hydsy=0
	for i in product(range(NBins),repeat=len(yds)+1):
		if pydsy[i]!=0 and pyds[i[1:]]!=0:
			hydsy = hydsy + pydsy[i]*log2(pyds[i[1:]]/pydsy[i])
	norm_factor = hydsy
	return TExy/norm_factor

def TExy(xd,yds,y,valor,NBins): #Transferencia de entropia de x para y
	"""
	"""
	pxdydsy, edges = histogramdd(vstack((xd,yds,y)).T, bins = NBins, normed=True,range = [valor[0]]+[valor[1]]*(len(yds)+1))
	pxdydsy = pxdydsy/sum(sum(sum(pxdydsy)))
	pxdyds, edges   = histogramdd(vstack((xd,yds)).T, bins = NBins, normed=True, range = [valor[0]]+[valor[1]]*len(yds))
	pxdyds = pxdyds/sum(pxdyds)
	pydsy, edges   = histogramdd(vstack((yds,y)).T, bins = NBins, normed=True, range = [valor[1]]*len(yds)+[valor[1]])
	pydsy = pydsy/sum(pydsy)
	if len(shape(yds))>1:
		pyds, edges   = histogramdd(yds.T, bins = NBins, normed=True, range = [valor[1]]*len(yds))
	else:
		pyds, edges   = histogramdd([yds.T], bins = NBins, normed=True, range = [valor[1]])
	pyds = pyds/sum(pyds)

	TExy = 0
	escala = range(NBins)
	
	# i - [xd, yds, y]
	for i in product(range(NBins),repeat=len(yds)+2):
		if pxdydsy[i]!=0 and pxdyds[i[:-1]]!=0 and pydsy[i[1:]]!=0 and pyds[i[1:-1]]!=0:
			TExy = TExy + pxdydsy[i]*(log2(pxdydsy[i]*pyds[i[1:-1]]/(pydsy[i[1:]]*pxdyds[i[:-1]])))
	return TExy

def DTE(x, y, Mlag, d, tau, valor, NBins):
	"""
	"""

	lag = range(Mlag)
	if Mlag > d*tau:
		NL  = len(x) - Mlag
	else:
		NL  = len(x) - d*tau

	yds=zeros((d,NL))
	for i in range(d):
		yds[i]=y[(Mlag-(i+1)*tau):(Mlag+NL-(i+1)*tau)]
	mec.push({'x': x})
	#mec.push({'y': y[Mlag:Mlag+NL]})
	mec.push({'y': y})
	#mec.push({'yds':yds})
	mec.push({'NL': NL})
	mec.push({'Mlag':  Mlag})
	mec.push({'valor': valor})
	mec.push({'NBins': NBins})
	mec.push({'TExy_norm':TExy_norm})
	
	mec.push({'d':d})
	mec.push({'tau':tau})

	mec.scatter('lag',lag)
	
	mec.execute('TE = [TExy_norm(x[Mlag-i:Mlag+NL-i], array([y[(Mlag-(j+1)*tau):(Mlag+NL-(j+1)*tau)].T for j in range(d)]), y[Mlag:Mlag+NL], valor, NBins) for i in lag]')
	DTE = mec.gather('TE')
	mec.results.clear()
	return [DTE, lag]

########################################### 
#            Prog principal               #
########################################### 

janela = 10000   # tamanho da janela de segmentacao em milisegundos
lag    =   105   # maximo lag em milisegundos
NBins  =   32   # numero de bins
NSur   =   10   # numero de surrogate

dados = open('dados.execucao', 'r')

for dado in dados:
	dado = dado[0:len(dado)-1]
	nome = dado.split()[0]
	trechos = [int(dado.split()[1]), int(dado.split()[2])]

	log = open(nome+'_TE.log', 'w')
	log.write('data da execucao: '+ str(datetime.now())+'\n')
	log.write('lag maximo DTExy: '+ str(lag)   +' ms\n')
	log.write('Num de bins    : '+str(NBins)+'\n')
	log.write('Num de NSurrog : '+str(NSur)+'\n')
	log.write(' ')

	if nome[-3:]=='ssa':
		Fs, Canais, ruim = SinaisLeitura.Canais(nome[:-4])
	else:
		Fs, Canais, ruim = SinaisLeitura.Canais(nome)
	log.write('Sample Frequency: '+ str(Fs)+'\n')
	log.write('Channels : '+ Canais[0]+' '+Canais[1]+' '+Canais[2]+' '+Canais[3]+'\n')
	canais = range(4) 
	if ruim == True:
		continue

	X = SinaisLeitura.sinais(nome, trechos[0], trechos[1]) 
	log.write('DataRange: '+ str(trechos[0])+'  '+str(trechos[1])+'\n')
	log.write('\n')
	
	print('finding embedding params')
	if os.path.isfile('%s.embedding'%nome):
		with open('%s.embedding'%nome,'r') as f:
			emb = dict(x.rstrip().split(None, 1) for x in f)
	else:
		print('calculating embedding params')
		emb={}
		for i in range(4):
			d,tau = SinaisMedidas.embedding_params(X[i],NBins)
			print('found %d,%d'%(tau,d))
			emb[Canais[i]]='%d %d'%(tau,d)
		print(emb)
		with open('%s.embedding'%nome, 'w') as f:
			for ch, vs in emb.items():
				print(ch,vs)
				f.write('%s %s'%(ch,vs) + '\n')
	
	EntSai  = permutations(canais, 2)
	for ch0, ch1 in EntSai:
		print('Channels: '+Canais[ch0] +' '+ Canais[ch1])
		if (0==0):  #'X' not in [Canais[ch0], Canais[ch1]]):
			x = X[ch0,:]
			y = X[ch1,:]
			tau=int(emb[Canais[ch1]].split()[0])
			d=int(emb[Canais[ch1]].split()[1])
			print('d: %d, tau: %d'%(d,tau))
			valor = [[min(x), max(x)], [min(y), max(y)]]
			log.write('ChannelsExec: '+str(Canais[ch0])+' '+str(Canais[ch1])+'\n')
			print('inicio DTE REF')
			tic = time.time()
			DTExyREF = DTE(x, y, int(Fs*lag/1000), d, tau, valor, NBins)
			tac = time.time()
			tempo= tac-tic
			print(tempo)
							
			st = str(nome)+'_'+ str(ch0)+'_'+ str(ch1) + '_REF.dte'
			dtereffile=open( st, "wb")
			pickle.dump( DTExyREF, dtereffile)
			dtereffile.close()
			
			log.write('Tempo Proc DTE Ref :'+str(time)+'\n')

			DTExySUR = []
			for j in range(NSur):
				tic = time.time()
				
				print('Obtaining surrogate %d'%j)
				xsur, msg = SinaisSurrogate.AAFT_fftw(x,0.001)
				print('Surrogate %d finished, starting DTE'%j)
				dteSUR = DTE(xsur, y, int(Fs*lag/1000), d, tau, valor, NBins)
				DTExySUR.append(dteSUR)
				tac = time.time()
				tempo= tac-tic
				print(tempo)
				log.write('Tempo Proc DTE Sur :'+str(tempo)+'\n')
			st = str(nome)+'_' + str(ch0)+'_'+ str(ch1)+'_SUR.dte'
			dtesurfile=open(st, "wb")
			pickle.dump( DTExySUR, dtesurfile)
			dtesurfile.close()
	log.write('END Processing')
	log.write('data de termino: '+ str(datetime.now())+'\n')
	log.close()
	del X
dados.close()
