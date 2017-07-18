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
# SinaisMedidas.py : Functions containing information-theoretic
# measures to apply on signals

from numpy import histogramdd,log2, corrcoef, array, zeros

def Ixy(x, y, valor, NBins): 
	"""Ixy(x, y) -> MI
			x e y sao sinais 
			valor = [[min(x), max(x)], [min(y), max(y)]]
			NBins eh o numero de bins usados para discretizar x e y
			essa funcao responde com a versao naive da estimativa da Informacao Mutua
	"""

	x = [float(i) for i in x]
	y = [float(i) for i in y]
	p, edges = histogramdd(x,     bins = NBins, range = [valor[0]])
	px = p/float(sum(p))
	p, edges = histogramdd(y,     bins = NBins, range = [valor[1]])
	py = p/float(sum(p))
	p, edges = histogramdd([x,y], bins = NBins, range = [valor[0],valor[1]])
	pxy = p/float(sum(sum(p)))
	MI = 0.0
	escala = range(NBins)
	for j in escala:
		for i in escala:
			if px[i]!=0 and py[j]!=0 and pxy[i,j]!=0:
				MI = MI + pxy[i,j]*(log2(pxy[i,j]/(px[i]*py[j])))
	return MI

def autocorr(x, t=1):
	#result = numpy.correlate(x, x, mode='full')
	#return result[result.size/2:]
	return corrcoef(array([x[0:len(x)-t], x[t:len(x)]]))[0,1]

def automi(x, t=1, NBins=32):
	NL = len(x) - 2*(t+1)
	return Ixy(x[t+1:t+1+NL],x[:NL],[[min(x), max(x)], [min(x), max(x)]],NBins)

def corr_mi(x,mtype='corr',NBins=32,start_tau=1,max_tau=1000, delta=1):
	if mtype=='corr':
		return array([autocorr(x,i) for i in range(start_tau,max_tau,delta)])
	else:
		return array([automi(x,i,NBins) for i in range(start_tau,max_tau,delta)])

def embedding_params(x,NBins=32,start_tau=1,max_tau=1000):
	d=1 # still find d
	tau=max_tau
	mi=1000
	for i in range(start_tau,max_tau):
		mifound = automi(x,i)
		#print mifound
		if mifound < mi:
			mi=mifound
		else:
			tau=i
			break
	if tau==max_tau:
		print("Warning: didn't find embedding delay")
	return d,tau

def teste_corrida(dados, h, alfa=0.05):
       """ Retorna o teste de corrida
            Para ser consideradas observacoes independentes, o 'num_corridas'
           calculado para as amostras deve estar no intervalo: [lim_inf, lim_sup]
            Para um nivel de significancia de 5%:
               z = 1.96(teste bilateral) e
               z = 1.65(teste unilateral)
            Vide site http://www.ufpa.br/dicas/biome/bionor.htm
       """
       n = len(dados)
       num_corridas = 1
       categorias = []
       categorias.append(dados[0] < h)
       for i in range(1, n):
           categorias.append(dados[i] < h)
           num_corridas += categorias[i] != categorias[i-1]
       z = abs(ss.norm.isf(1 - alfa/2.0))
       n0 = categorias.count(0)
       n1 = n - n0
       dnn = 2*n0*n1
       mi_r = dnn/float(n) + 1
       dp_r = sqrt(dnn * (dnn - n) / ((n-1)*n**2))
       lim_inf = mi_r - z*dp_r
       lim_sup = mi_r + z*dp_r
       return lim_inf <= num_corridas <= lim_sup

def teste_tendencia(dados, alfa=0.05):
       """ Retorna o teste de tendencia
            Para ser consideradas observacoes independentes, o valor de 'A'
           calculado para as amostras deve estar no intervalo: [lim_inf, lim_sup]
            Para um nivel de significancia de 5%:
               z = 1.96(teste bilateral) e
            Vide site http://www.ufpa.br/dicas/biome/bionor.htm
       """
       n = len(dados)
       A = 0.0
       for i in range(n-1):
           hm = 0
           j = i + 1
           while j < n:
               hm += dados[i] > dados[j]
               j += 1
           A += hm
       mi_A = n*(n-1) / 4.0
       dp_A = sqrt(n*(2*n+5) * (n-1)/72.0)
       z = abs(ss.norm.isf(1 - alfa/2.0))
       lim_inf = mi_A - z*dp_A
       lim_sup = mi_A + z*dp_A
       printd("\nTeste de tendencia", DP)
       printd("A: {0}".format(A), DP)
       msg = "  {0:.2f} <= A <= {1:.2f}"
       printd(msg.format(lim_inf, lim_sup), DP)
       return lim_inf <= A <= lim_sup

def eh_estacionarios(dados, alfa=0.05):  #limites_segmento,
       """
           limites_segmento = [a, b]
       """
       segmento = dados     
       dados = signal.detrend(dados)
       segmentos = split(segmento, 2**5)
       segmentos = array(segmentos)
       medias = segmentos.mean(1)
       variancias = segmentos.var(1)
       media_h = segmentos.mean()
       variancia_h = segmentos.var()
       return self.teste_corrida(medias, media_h, alfa) \
              and self.teste_corrida(variancias, variancia_h, alfa)
