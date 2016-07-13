#!/usr/bin/python
# coding: utf-8
  
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
import numpy as np
import matplotlib.pyplot as plt


with open("C:\Users\Valentin\Desktop\Stage M2\liste_traite.txt", "r") as f :
	
	fichier_entier = f.read()
	files = fichier_entier.split("\n")


i = 1
for fichier in files :
	
	with open(fichier, 'r') :
 
                outname = fichier.replace('traitement_1', 'dereddenedXY')

		print "Traitement du fichier : " + str(i) + '/' + str(len(files))  
		reading = fits.open(fichier)           # Ouverture du fichier à l'aide d'astropy
 
		tbdata = reading[1].data               # Lecture des données fits
            
            #################################################
            # Détermination des valeurs extremales du champ #
            #################################################

		# Détermination de RA_max et RA_min 
		RA_max = np.max(tbdata['RA'])
		RA_min = np.min(tbdata['RA'])
		print "RA_max vaut :     " + str(RA_max)
		print "RA_min vaut :     " + str(RA_min)

		# Détermination de DEC_max et DEC_min	
		DEC_max = np.max(tbdata['DEC'])
		DEC_min = np.min(tbdata['DEC'])
		print "DEC_max vaut :   " + str(DEC_max)
		print "DEC_min vaut :   " + str(DEC_min)
 
            #########################################
            # Calcul de la valeur centrale du champ #
            #########################################
 
		# Détermination de RA_moyen et DEC_moyen
		RA_central = (RA_max + RA_min)/2.
		DEC_central = (DEC_max + DEC_min)/2.
 
		print "RA_central vaut : " + str(RA_central)
		print "DEC_central vaut : " + str(DEC_central)
	
		print " "
		

	    #############################################################################
	    # Détermination de X et de Y et création d'un nouveau fits + dérougissement #
	    #############################################################################

		Ag = 3.237
		Ar = 2.176
		Ai = 1.595
		Az = 1.217

		# Creation du tableau
		new_col_data_X = array = (tbdata['RA'] - RA_central) * np.cos(DEC_central)
		new_col_data_Y = array = tbdata['DEC'] - DEC_central
		new_col_data_g0 = array = tbdata['G'] - tbdata['EBV'] * Ag
		new_col_data_r0 = array = tbdata['R'] - tbdata['EBV'] * Ar
		new_col_data_i0 = array = tbdata['I'] - tbdata['EBV'] * Ai
		new_col_data_z0 = array = tbdata['Z'] - tbdata['EBV'] * Az

                print "Création du tableau"

		# Creation des nouvelles colonnes
		col_X = fits.Column(name='X', format='D', array=new_col_data_X)
		col_Y = fits.Column(name='Y', format='D', array=new_col_data_Y)
		col_g0 = fits.Column(name='g0', format='D', array=new_col_data_g0)
		col_r0 = fits.Column(name='r0', format='D', array=new_col_data_r0)
		col_i0 = fits.Column(name='i0', format='D', array=new_col_data_i0)
		col_z0 = fits.Column(name='z0', format='D', array=new_col_data_z0)

		# Creation de la nouvelle table
		tbdata_new = fits.BinTableHDU.from_columns(tbdata.columns + col_g0 + col_r0 + col_i0 + col_z0 + col_X + col_Y)

		# Ecriture du fichier de sortie .fits
		tbdata_new.writeto(outname)

                i += 1
                print "Ecriture du nouveau fichier"

                print " "
		print " ------------------------------- "
		print " "

reading.close()
