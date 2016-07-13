#!/usr/bin/python
# coding: utf-8
 
from astropy.io import fits
import numpy as np
 
            #########################################
            # Fichier contenant la liste des champs #
            #########################################


with open("E:\New_Fields\liste_essai.txt", "r") as f :
	
	fichier_entier = f.read()
	files = fichier_entier.split("\n")

i = 1

for fichier in files :
	
	with open(fichier, 'r') :
 
            outname = fichier.replace('combined_final_roughcal', 'traitement_1')
    
            print "Traitement du fichier : " + str(i) + '/' + str(len(files))  
            reading = fits.open(fichier,memmap = True)           # Ouverture du fichier à l'aide d'astropy
            
            tbdata = reading[1].data               # Lecture des données fits
            
    
   	    #######################################################
                # Application du tri en fonction de divers paramètres #
                #######################################################
    
  	 # Création d'un masque pour la condition CHI
            mask1 = tbdata['CHI'] < 1.4	
            tbdata = tbdata[mask1]
            
           # print "Tri effectué sur CHI"
            
            # Création d'un 3e masque sur la condition SHARP (1/2)
            mask2 = tbdata['SHARP'] > -0.35    
            tbdata = tbdata[mask2]
            
            mask3 = tbdata['SHARP'] < 0.1
            tbdata = tbdata[mask3]
            
         #   print "Tri effectué sur SHARP"
        
            # Création d'un premier masque sur la condition PROB
            mask4 = tbdata['PROB'] > 0.01  
            tbdata = tbdata[mask4]
        
            mask5 = tbdata['PROB'] < 1.01
            tbdata = tbdata[mask5]
            
          #  print "Tri effectué sur PROB"
            
          		###################################################
          		# Ecriture du résultat dans nouveau fichier .fits #
          		###################################################
            
               	
            hdu = fits.BinTableHDU(data=tbdata)
            ecriture = hdu.writeto(outname) 
            
            print "Ecriture du nouveau fichier traité"
            print "-------------------------- "
            
          		######################################################
          		# Suppression des variables pour redonner la mémoire #
          		######################################################
            
            i += 1

            del tbdata
            del mask1, mask2, mask3, mask4, mask5        