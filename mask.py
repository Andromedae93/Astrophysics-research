# -*- coding: utf-8 -*-
#!/usr/bin/env python
  
from astropy.io import fits
import numpy as np
    
                ###################################
                # Fichier contenant le champ brut #
                ###################################

with open("E:\New_Fields\liste_essai.txt", "r") as f :
	
	fichier_entier = f.read()
	files = fichier_entier.split("\n")

i = 1
for fichier in files :
   	
	with open(fichier, 'r') :
 
            outname = fichier.replace('combined_final_roughcal', 'mask')

            # Ouverture du fichier à l'aide d'astropy  
            field = fits.open(fichier, 'readonly', memmap=True)   
            print "Traitement du fichier : " + str(i) + '/' + str(len(files))     
            print " "  
    
            # Lecture des données fits
            tbdata = field[1].data   
        #   print "Lecture des données du fits"            
    
                            ###############################
                            # Application du tri sur PROB #
                            ###############################
    
            mask = np.bitwise_and(tbdata['PROB'] < 1.1, tbdata['PROB'] > -0.1)  
            new_tbdata = tbdata[mask]   
            #print "Création du Masque"       
            #print " "
    
                        #################################################
                        # Détermination des valeurs extremales du champ #
                        #################################################
    
            # Détermination de RA_max et RA_min 
            RA_max = np.max(new_tbdata['RA'])
            RA_min = np.min(new_tbdata['RA'])
            #print "RA_max vaut :     " + str(RA_max)
            #print "RA_min vaut :     " + str(RA_min)
    
            # Détermination de DEC_max et DEC_min   
            DEC_max = np.max(new_tbdata['DEC'])
            DEC_min = np.min(new_tbdata['DEC'])
            #print "DEC_max vaut :   " + str(DEC_max)
            #print "DEC_min vaut :   " + str(DEC_min)
    
                        #########################################
                        # Calcul de la valeur centrale du champ #
                        #########################################
    
            # Détermination de RA_moyen et DEC_moyen
            RA_central = (RA_max + RA_min)/2.
            DEC_central = (DEC_max + DEC_min)/2.
    
    #        print "RA_central vaut : " + str(RA_central)
    #        print "DEC_central vaut : " + str(DEC_central)
    #
    #        print " "
    #        print " ------------------------------- "
    #        print " "
    
                    ##############################
                    # Détermination de X et de Y #
                    ##############################
    
    
            # Creation du tableau
            new_col_data_X = array = (new_tbdata['RA'] - RA_central) * np.cos(DEC_central)
            new_col_data_Y = array = new_tbdata['DEC'] - DEC_central
        #  print 'Création du tableau'
    
    
            # Creation des nouvelles colonnes
            col_X = fits.Column(name='X', format='D', array=new_col_data_X)
            col_Y = fits.Column(name='Y', format='D', array=new_col_data_Y)
            #print 'Création des nouvelles colonnes X et Y'
    
    
            # Creation de la nouvelle table
            tbdata_final = fits.BinTableHDU.from_columns(new_tbdata.columns + col_X + col_Y)
    
            # Ecriture du fichier de sortie .fits
            tbdata_final.writeto(outname)
            print 'Ecriture du fichier termine'
    
            del field, tbdata, mask, new_tbdata, new_col_data_X, new_col_data_Y, col_X, col_Y, tbdata_final
    
            i = i + 1
            print " "
            print " ......................................................................................"
            print " "