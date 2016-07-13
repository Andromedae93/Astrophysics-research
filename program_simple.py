#!/usr/bin/python
# coding: utf-8
  
from astropy.io import fits
from astropy.table import Table
import numpy as np
import time

start = time.time()
  
		        ###################################
                # Fichier contenant le champ brut #
                ###################################
  
filename = 'E:/New_Fields/Field138_combined_final_roughcal.fits'

outname = filename.replace('combined_final_roughcal', 'traitement_1')
outname2 = filename.replace('combined_final_roughcal', 'test')

# Ouverture du fichier à l'aide d'astropy  
with fits.open(filename) as field :          
  
    print 'Fichier en cours de traitement' + str(filename) + '\n'
    print " "
    
    # Lecture des données fits
    tbdata = field[1].data               
    
    print'Donnees lues'
    
   	        #######################################################
                    # Application du tri en fonction de divers paramètres #
                    #######################################################
    
    # Création d'un masque pour la condition CHI
    mask1 = tbdata['CHI'] < 1.4	
    tbdata = tbdata[mask1]
     
    print "Tri effectué sur CHI"
    
    # Création d'un 3e masque sur la condition SHARP (1/2)
    mask2 = tbdata['SHARP'] > -0.35    
    tbdata = tbdata[mask2]
    
    mask3 = tbdata['SHARP'] < 0.1
    tbdata = tbdata[mask3]
    
    print "Tri effectué sur SHARP"

    # Création d'un premier masque sur la condition PROB
    mask4 = tbdata['PROB'] > 0.01  
    tbdata = tbdata[mask4]

    mask5 = tbdata['PROB'] < 1.01
    tbdata = tbdata[mask5]
    
    print "Tri effectué sur PROB"
    
  		###################################################
  		# Ecriture du résultat dans nouveau fichier .fits #
  		###################################################
    
       	
    hdu = fits.BinTableHDU(data=tbdata)
    ecriture = hdu.writeto(outname) 
    
    print "Ecriture du nouveau fichier traité"
    
  		######################################################
  		# Suppression des variables pour redonner la mémoire #
  		######################################################
    
    
    del tbdata
    del mask1
    del mask2
    del mask3	
    
    end = time.time()
    
    print (end-start)/60