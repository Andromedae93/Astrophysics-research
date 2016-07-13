#!/usr/bin/python
# coding: utf-8
  
from astropy.io import fits
from astropy.table import Table
import numpy as np
import time

start = time.time()
  
		################################
                # File which contains raw data #
                ################################
  
filename = 'E:/New_Fields/Field138_combined_final_roughcal.fits'

outname = filename.replace('combined_final_roughcal', 'traitement_1')
outname2 = filename.replace('combined_final_roughcal', 'test')

# Open the file with Astropy 
with fits.open(filename) as field :          
  
    print 'Fichier en cours de traitement' + str(filename) + '\n'
    print " "
    
    # Data fits reading
    tbdata = field[1].data               
    
    print'Donnees lues'
    
   	            ####################################
                    # Selection about some parameters  #
                    ####################################
    
    # Mask creation to 'CHI SQUARE' condition
    mask1 = tbdata['CHI'] < 1.4	
    tbdata = tbdata[mask1]
     
    print "Tri effectué sur CHI"
    
    # Mask creation to 'SHARP' condition
    mask2 = tbdata['SHARP'] > -0.35    
    tbdata = tbdata[mask2]
    
    mask3 = tbdata['SHARP'] < 0.1
    tbdata = tbdata[mask3]
    
    print "Tri effectué sur SHARP"

    # Mask creation to 'PROB' condition
    mask4 = tbdata['PROB'] > 0.01  
    tbdata = tbdata[mask4]

    mask5 = tbdata['PROB'] < 1.01
    tbdata = tbdata[mask5]
    
    print "Tri effectué sur PROB"
    
  		########################################
  		# Write the result in a new .fits file #
  		########################################
    
       	
    hdu = fits.BinTableHDU(data=tbdata)
    ecriture = hdu.writeto(outname) 
    
    print "Ecriture du nouveau fichier traité"
    
  		###################################
  		# Remove variables to free memory #
  		###################################
    
    
    del tbdata
    del mask1
    del mask2
    del mask3	
    
    end = time.time()
    
    print (end-start)/60
