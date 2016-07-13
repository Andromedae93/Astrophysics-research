#!/usr/bin/python
# coding: utf-8
  
from astropy.io import fits
import numpy as np


            ##################################
            # Fichier contenant le champ 169 #
            ##################################
  
filename ="E:\Fields\Field247_traitement_1.fits"

outname = filename.replace('traitement_1', 'dereddenedXY')

print 'Fichier en cours de traitement' + str(filename) + '\n'

# Ouverture du fichier à l'aide d'astropy  
field = fits.open(filename)          
  
# Lecture des données fits
tbdata = field[1].data             


            #################################################
            # Détermination des valeurs extremales du champ #
            #################################################

# Détermination de RA_max et RA_min 
RA_max = np.max(tbdata['RA'])
print tbdata['RA']
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
print " ------------------------------- "
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

field.close()