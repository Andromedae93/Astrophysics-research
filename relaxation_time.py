#!/usr/bin/python
# coding: utf-8

import math

############################
# Definition des variables #
############################
M_sun = 2.*10**30
M_cluster = (10.**3)				# 10³ masses solaires
M_LMC = (10.**10)				# Masse LMC : 10¹⁰ masses solaires
R_hl = 10*(3.085*10**16)			# Rayon demi-lumière du cluster : 10 pc en mètres
R_gal = 10700.					# Distance LMC / cluster = 10.7 kpc
m_star = 0.6
G = 6.67*10**(-11)
log_Coulomb = 0.4				# Incertain

Gyear = 31536000*(10**9)			# 1 Gy en seconde

############################
# Définition des grandeurs #
############################

def relaxation_time (M_cluster, R_hl, m_star, G, log_Coulomb) :
	
	return 0.14 * (M_cluster**(1/2)) * (R_hl**(3/2)) / (m_star**(1/2) * (G**(1/2)) * math.log(0.4*500))


def tidal_radius (M_cluster, M_LMC, R_gal) :

	return 0.43 * R_gal * (M_cluster/(M_LMC))**(1./3.)

###########################
# Affichage des grandeurs #
###########################
#X = R_gal*10**(7./3.)

#print

print "Temps de relaxation : " + str(relaxation_time (M_cluster, R_hl, m_star, G, log_Coulomb)) + "s soit " + str((relaxation_time (M_cluster, R_hl, m_star, G, log_Coulomb))/Gyear) + " Gy"

print "Rayon de marée : " + str(tidal_radius (M_cluster, M_LMC, R_gal)) + " pc"


##################################
# Calcul du temps de dissolution #
##################################

Beta = 1.91
N = 10**4
Gamma = 0.02
x = 0.75
R_g = 10.7
V_g = 186
eps = 0

def dissolution_time (Beta,N,Gamma,x,R_g,V_g,eps) :

	return Beta*(N/math.log(Gamma*N))**x * R_g * (V_g)**-1 * (1-eps)

print "Le temps de dissolution de notre cluster vaut : " + str(dissolution_time(Beta,N,Gamma,x,R_g,V_g,eps)) + " Myr"



