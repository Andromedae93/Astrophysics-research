# -*- coding: utf-8 -*-
#!/usr/bin/env python
  
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
import numpy as np
import scipy.ndimage as sp
import matplotlib.pyplot as plt
#import matplotlib.figure as mfig
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.set_printoptions(threshold=np.nan)

#plt.viridis()   # Permet d'avoir des couleurs différentes pour les graphiques

	    ###################################
	    # Importation du fichier de champ #
	    ###################################

filename = '/Users/jungbluth/Desktop/Stage/Data/Champ 169/Field169_dereddenedXY.fits'

print 'Fichier en cours de traitement' + str(filename) + '\n'
print " "


test =  filename.split('/')
test2 =  test[6].split('_')
print test2
fieldname = test2[0] # Récupère le Field en traitement
print fieldname


                            ##################################################################
                            #                                                                #
                            #   Création des noms de sorties de chaques graphiques & masks   #
                            #                                                                #
                            ##################################################################

outname = filename.replace('dereddenedXY.fits', 'Distribution_etoiles.png')
outname2 = filename.replace('dereddenedXY.fits', 'Carte_densite_nonlisse.png')
outname3 = filename.replace('dereddenedXY.fits', 'Carte_densite_lisse.png')
outname6 = filename.replace('dereddenedXY.fits', 'Histogramme2D_heatmap.png')
outname7 = filename.replace('dereddenedXY.fits', 'Masque_brut.png')
outname8 = filename.replace('dereddenedXY.fits', 'Masque_convolue.png')
outname10 = filename.replace('dereddenedXY.fits', 'SN_map.png')
outname11 = filename.replace('dereddenedXY.fits', 'Histogramme_SN_map.png')
outname12 = filename.replace('dereddenedXY.fits', 'SN_map_final.png')
outname15 = filename.replace('dereddenedXY.fits', 'resume.png')

  
# Ouverture du fichier à l'aide d'astropy 
field = fits.open(filename)         

# Lecture des données fits  
tbdata = field[1].data             

                            ##############################################################################
                            #                                                                            #
                            # Parametres pour la carte de distribution des étoiles bleues et très bleues #
                            # Création de boîtes de sélections en g-r et g                               #
                            #                                                                            #
                            ##############################################################################

# Boite des étoiles bleues :
condition_1 = np.bitwise_and( (tbdata['g0'] - tbdata['r0']) > 0, (tbdata['g0'] - tbdata['r0']) < 0.8 )	# Ne garder que les -0.4 < (g-r)0 < 0.8
condition_final = np.bitwise_and(tbdata['g0'] < 23.5, condition_1)		# Récupere les valeurs de 'g0' < 23.5 dans les valeurs de blue_stars_X

Blue_stars = tbdata[condition_final]

RA_Blue_stars = Blue_stars['RA']						# Récupere les valeurs de 'RA' associées aux étoiles bleues
DEC_Blue_stars = Blue_stars['DEC']						# Récupere les valeurs de 'DEC' associées aux étoiles bleues

print "Création de la sélection d'étoiles bleues"

# Boite des étoiles tres bleues :
condition_2 = np.bitwise_and( (tbdata['g0'] - tbdata['r0']) > -0.5, (tbdata['g0'] - tbdata['r0']) < 0 )
condition_final2 = np.bitwise_and(tbdata['g0'] < 23.5, condition_2)

Very_Blue_stars = tbdata[condition_final2]

RA_Very_Blue_stars = Very_Blue_stars['RA']						# Récupere les valeurs de 'RA' associées aux étoiles bleues
DEC_Very_Blue_stars = Very_Blue_stars['DEC']

print "Création de la sélection d'étoiles très bleues"

# ==> La table finale avec le masque s'appelle Blue_stars & Very_Blue_stars

                            ####################################################################
                            #                                                                                                                          #
                            #   Traçage des différents graphiques de la distribution d'étoiles                                #
                            #   Représentation identiques à Topcat sous forme de 4 graphiques                           #
                            #                                                                                                                          #
                            ####################################################################

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig = plt.gcf()
fig.set_size_inches(16, 9)


ax1.plot(tbdata['g0'] - tbdata['r0'], tbdata['g0'], 'k.', label=u'Etoiles du champ', markersize=1)
ax1.plot( (Blue_stars['g0']-Blue_stars['r0']), Blue_stars['g0'], 'b.', label =u'Etoiles bleues', markersize=1)
ax1.plot( (Very_Blue_stars['g0']-Very_Blue_stars['r0']), Very_Blue_stars['g0'], 'r.', label =u'Etoiles tres bleues', markersize=1)
ax1.set_title('Diagramme Couleur-Magnitude')
ax1.set_xlabel(r'$(g-r)_0$')
ax1.set_ylabel(r'$g_0$')
ax1.set_xlim(-1.5,2.5)
ax1.set_ylim(14,28)
ax1.legend(loc='upper left', frameon=None)
ax1.invert_yaxis()

ax2.plot(RA_Blue_stars, DEC_Blue_stars, 'b.', label =u'Etoiles bleues', alpha=1, markersize=1)
ax2.set_title("Carte de distribution des etoiles bleues")
ax2.set_xlabel('RA')
ax2.set_ylabel('DEC')
ax2.legend(loc='upper left')

ax3.plot(RA_Very_Blue_stars, DEC_Very_Blue_stars, 'r.', label =u'Etoiles tres bleues', alpha =1)
ax3.set_title("Carte de distribution des etoiles tres bleues")
ax3.set_xlabel('RA')
ax3.set_ylabel('DEC')
ax3.legend(loc='upper left')
	
ax4.plot(RA_Blue_stars, DEC_Blue_stars, 'b.', label =u'Etoiles bleues',markersize=1)
ax4.plot(RA_Very_Blue_stars, DEC_Very_Blue_stars, 'r.', label =u'Etoiles tres bleues')
ax4.set_xlabel('RA')
ax4.set_ylabel('DEC')
ax4.legend(loc='upper left')

fig1.tight_layout()

fig1.savefig(outname)

print "Création des cartes de visualisation identiques à Topcat"

'''

		######################################################################
		# Traçage des différents graphiques de la carte de densité d'étoiles #
		######################################################################


# Carte de densité des étoiles bleues pour 1 pixel de 1 arcmin^2 (bins = 180)

X_Blue_stars = Blue_stars['X']
Y_Blue_stars = Blue_stars['Y']
			
heatmap, xedges, yedges = np.histogram2d(X_Blue_stars, Y_Blue_stars, bins=[265,241]) # y = 2° sur x = 2.18° ce qui fait : 0.5'x0.5'
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
		
RotatePlot_old = sp.rotate(heatmap,270)
RotatePlot = np.fliplr(RotatePlot_old)

fig2 = plt.gcf()
fig2.set_size_inches(16, 9)

fig2 = plt.figure()
fig_heatmap = plt.imshow(RotatePlot, interpolation='nearest')
cbar = plt.colorbar()
cbar.set_label(r"Densite moyenne d'etoiles bleues par pixel")
plt.title("Distribution des etoiles bleues (pixel = 0,5' x 0,5')")
plt.xlabel("pixel i")
plt.ylabel("pixel j")
plt.gca().invert_yaxis()


fig2 = plt.savefig(outname2)


                                #######################################################################
                                #                                                                     #
                                #   Création de la carte de densité des étoiles bleues convoluée      #
                                #   Heatmap convoluée pour 2'                                         #
                                #   Heatmap convoluée pour 8'                                         #
                                #                                                                     #    
                                #######################################################################
        
fig3, (ax1, ax2, ax3) = plt.subplots(1,3)

fig = plt.gcf()
fig.set_size_inches(16, 9)

convolution_locale = convolve(RotatePlot, Gaussian2DKernel(stddev=4)) # AFFICHAGE DE LA CARTE DE DENSITE CONVOLUEE POUR 2'
fig_smoothed_heatmap_locale = ax1.imshow(convolution_locale, interpolation='nearest')
ax1.set_title("Carte de densite convoluee 2'")
ax1.set_xlabel("X (arcmin)")
ax1.set_ylabel("Y (arcmin)")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig3.colorbar(fig_smoothed_heatmap_locale,cax=cax1)
ax1.invert_yaxis()


convolution_grande = convolve(RotatePlot, Gaussian2DKernel(stddev=32)) # AFFICHAGE DE LA CARTE DE DENSITE CONVOLUEE POUR 8'
fig_smoothed_heatmap_grande = ax2.imshow(convolution_grande, interpolation='nearest')
ax2.set_title("Carte de densite convoluee 16'")
ax2.set_xlabel("X (arcmin)")
ax2.set_ylabel("Y (arcmin)")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig3.colorbar(fig_smoothed_heatmap_grande,cax=cax2)
ax2.invert_yaxis()

convolution_diff = convolution_locale - convolution_grande # AFFICHAGE DE LA CARTE DE DENSITE CONVOLUEE 2' - 8'
fig_smoothed_tab_diff = ax3.imshow(convolution_diff, interpolation='nearest')
ax3.set_title("Carte 2' - Carte 16'")
ax3.set_xlabel("X (arcmin)")
ax3.set_ylabel("Y (arcmin)")
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig3.colorbar(fig_smoothed_tab_diff,cax=cax3)
ax3.invert_yaxis()

# Create space for labels between subplots
fig3.tight_layout()

fig3.savefig(outname3)

print "Création de la carte de surdensité : comptage - background"
print " "


		#############################
		# Histogramme 1D de heatmap #
		#############################


tab = heatmap.flatten()
fig4 = plt.figure(4)
fig_tab = plt.hist(tab, bins = 11, range=(-0.5,10.5))

tab_diff = convolution_diff.flatten()
fig5 = plt.figure(5)
fig_tab_diff = plt.hist(tab_diff, bins = 50, range=(-1,1))


		#####################################
		# Histogramme 2D de la carte lissée #
		#####################################


smoothed_tab_locale = convolution_locale.flatten() 		# Devient une table 1D

smoothed_tab_grande = convolution_grande.flatten()

new_smoothed_tab_locale = smoothed_tab_locale[smoothed_tab_locale > 0.45]	# Création de la nouvelle smoothed_tab permettant de ne garder que les valeurs > 0.5 (retire le bruit)
new_smoothed_tab_grande = smoothed_tab_grande[smoothed_tab_grande > 0.45]

fig = plt.gcf()
fig.set_size_inches(12, 8)

fig6 = plt.subplot(1,2,1)
result_locale = plt.hist(new_smoothed_tab_locale, bins = 500, normed = False, range=(0,2))	#Traçage de l'histogramme a 2'
plt.xlabel("Number of stars per pixel")
plt.ylabel("Number of pixels normalized")
plt.title("Histogram from 2' convolution")

fig6 = plt.subplot(1,2,2)
result_grande = plt.hist(new_smoothed_tab_grande, bins = 500, normed = False, range=(0,3))	#Traçage de l'histogramme a 8'
plt.xlabel("Number of stars per pixel")
plt.ylabel("Number of pixels normalized")
plt.title("Histogram from 8' convolution")

fig6 = plt.savefig(outname6)

		######################################################
		# Gaussienne ajustée à l'histogramme 2D à sttdev = 2 #
		######################################################	

plt.xlim(min(new_smoothed_tab_locale), max(new_smoothed_tab_locale))			# Détermination des composantes de la Gaussienne

mean_locale = np.mean(new_smoothed_tab_locale)
variance_locale = np.var(new_smoothed_tab_locale)
sigma_locale = np.sqrt(variance_locale)

x_locale = np.linspace(min(new_smoothed_tab_locale), max(new_smoothed_tab_locale), 22807)

dx_locale = result_locale[1][1] - result_locale[1][0]
scale_locale = len(new_smoothed_tab_locale)*dx_locale
Gaussian_locale =  mlab.normpdf(x_locale,mean_locale,sigma_locale)*scale_locale
fig_Gaussian_locale = plt.plot(x_locale,Gaussian_locale)	# Traçage de la Gaussienne ajustée à la distribution (permet de ne pas normaliser l'histogramme)

print " "
print "Pour une Gaussienne de 1' :"
print "La valeur de la moyenne est : " + str(mean_locale)
print "La valeur de la variance est : " + str(variance_locale)
print "La valeur de sigma est : " + str(sigma_locale)
print " "


		######################################################
		# Gaussienne ajustée à l'histogramme 2D à sttdev = 8 #
		######################################################	

plt.xlim(min(new_smoothed_tab_grande), max(new_smoothed_tab_grande))			# Détermination des composantes de la Gaussienne

mean_grande = np.mean(new_smoothed_tab_grande)
variance_grande = np.var(new_smoothed_tab_grande)
sigma_grande = np.sqrt(variance_grande)

x_grande = np.linspace(min(new_smoothed_tab_grande), max(new_smoothed_tab_grande), 22807)

dx_grande = result_grande[1][1] - result_grande[1][0]
scale_grande = len(new_smoothed_tab_grande)*dx_grande
Gaussian_grande =  mlab.normpdf(x_grande,mean_grande,sigma_grande)*scale_grande
fig_Gaussian_grande = plt.plot(x_grande,Gaussian_grande)	# Traçage de la Gaussienne ajustée à la distribution (permet de ne pas normaliser l'histogramme)

print "Pour une Gaussienne de 4' :"
print "La valeur de la moyenne est : " + str(mean_grande)
print "La valeur de la variance est : " + str(variance_grande)
print "La valeur de sigma est : " + str(sigma_grande)



                                #######################################################################################
                                #                                                                                     #
                                #   Création du fichier mask                                                          #
                                #   Tri effectué sur PROB pour ne garder que les valeurs comprises entre -0.1 et 1.1  #
                                #   Détermination des valeurs extrémales puis centrales du champ                      #
                                #   Détermination de X et Y                                                           #
                                #                                                                                     #
                             #######################################################################################


                ###################################
                # Fichier contenant le champ brut #
                ###################################

filename =  '/Users/jungbluth/Desktop/Stage/Data/Champ 169/Field169_combined_final_roughcal.fits'    

outname2 = filename.replace('combined_final_roughcal', 'mask')

# Ouverture du fichier à l'aide d'astropy  
field = fits.open(filename)   
print "Ouverture du fichier : " + str(filename)       

# Lecture des données fits
tbdata = field[1].data   
print "Lecture des données du fits"            

                ###############################
                # Application du tri sur PROB #
                ###############################

mask = np.bitwise_and(tbdata['PROB'] < 1.1, tbdata['PROB'] > -0.1)  
new_tbdata = tbdata[mask]   
print "Création du Masque"       
print " "

            #################################################
            # Détermination des valeurs extremales du champ #
            #################################################

# Détermination de RA_max et RA_min 
RA_max = np.max(new_tbdata['RA'])
RA_min = np.min(new_tbdata['RA'])
print "RA_max vaut :     " + str(RA_max)
print "RA_min vaut :     " + str(RA_min)

# Détermination de DEC_max et DEC_min   
DEC_max = np.max(new_tbdata['DEC'])
DEC_min = np.min(new_tbdata['DEC'])
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

        ##############################
        # Détermination de X et de Y #
        ##############################


# Creation du tableau
new_col_data_X = array = (new_tbdata['RA'] - RA_central) * np.cos(DEC_central)
new_col_data_Y = array = new_tbdata['DEC'] - DEC_central
print 'Création du tableau'


# Creation des nouvelles colonnes
col_X = fits.Column(name='X', format='D', array=new_col_data_X)
col_Y = fits.Column(name='Y', format='D', array=new_col_data_Y)
print 'Création des nouvelles colonnes X et Y'


# Creation de la nouvelle table
tbdata_final = fits.BinTableHDU.from_columns(new_tbdata.columns + col_X + col_Y)

# Ecriture du fichier de sortie .fits
tbdata_final.writeto(outname2)
print 'Ecriture du nouveau fichier mask'


field.close()

                                ##################################################################################################
                                #                                                                                                #
                                # Création de la grille permettant de déterminer si il y a ou non des étoiles dans chaque pixels #
                                # Si présence d'étoiles = 1, si absence d'étoiles = 0                                            #
                                #                                                                                                #
                                ##################################################################################################

filename =  '/Users/jungbluth/Desktop/Stage/Data/Champ 169/Field169_mask.fits'

print ' '
print 'Fichier en cours de traitement' + str(filename) + '\n'

# Opening file with astropy
field = fits.open(filename)       

# fits data reading 
tbdata = field[1].data    

##### BUILDING A GRID FOR THE DATA ########
nodesx,nodesy = 264,240   # PIXELS IN X, PIXELS IN Y
firstx,firsty = np.min(tbdata['X']),np.min(tbdata['Y'])
sizex = (np.max(tbdata['X'])-np.min(tbdata['X']))/nodesx
sizey = (np.max(tbdata['Y'])-np.min(tbdata['Y']))/nodesy
grid = np.zeros((nodesx+1,nodesy+1),dtype='bool') # PLUS 1 TO ENSURE ALL DATA IS INSIDE GRID

# CALCULATING GRID COORDINATES OF DATA
indx = np.int_((tbdata['X']-firstx)/sizex)
indy = np.int_((tbdata['Y']-firsty)/sizey)
grid[indx,indy] = True  # WHERE DATA EXISTS SET TRUE

# PLOT MY FINAL IMAGE
fig7 = plt.figure(7)
fig_grid = plt.imshow(grid.T,origin='lower',cmap='binary',interpolation='nearest')
plt.title('Masque du champ 169')

binary_mask = grid.T



fig7.savefig(outname7)

                                #######################################################################
                                #                                                                     #
                                # Création des masques convolués par une Gaussienne de 2' et 8'       #
                                # Puis rotation à 180° des résultats obtenus pour garder le bon sens  #
                                #                                                                     #
                                #######################################################################

fig8, (ax1, ax2) = plt.subplots(1,2)  # Affichage du masque convolué pour 2' et 8'

fig = plt.gcf()
fig.set_size_inches(16, 9)

convolution_mask_locale = convolve(binary_mask, Gaussian2DKernel(stddev=4),boundary='extend')   # Affichage du masque convolué pour 2'
#convolution_mask_locale = sp.rotate(convolution_mask_locale_old,180)    # Rotation à 180° de notre masque convolué
fig_convolution_mask_locale = ax1.imshow(convolution_mask_locale, interpolation='nearest')
ax1.set_xlabel("X (arcmin)")
ax1.set_ylabel("Y (arcmin)")
ax1.set_title("Masque convolue 2'")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig8.colorbar(fig_convolution_mask_locale,cax=cax1)
ax1.invert_yaxis()


convolution_mask_grande = convolve(binary_mask, Gaussian2DKernel(stddev=32),boundary='extend')   # Affichage du masque convolué pour 8'
#convolution_mask_grande = sp.rotate(convolution_mask_grande_old,180)    # Rotation à 180° de notre masque convolué
fig_convolution_mask_grande = ax2.imshow(convolution_mask_grande, interpolation='nearest')
ax2.set_xlabel("X (arcmin)")
ax2.set_ylabel("Y (arcmin)")
ax2.set_title("Masque convolue 16'")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig8.colorbar(fig_convolution_mask_grande,cax=cax2)
ax2.invert_yaxis()


fig8.tight_layout()

fig8.savefig(outname8)


                                #######################################################################
                                #                                                                     #
                                #     Création de 3 graphiques permettant de donner une carte S/N     #
                                #     Graphique 1 : Heatmap convoluée à 2' - Masque convolué à 2'     #
                                #     Graphique 2 : Heatmap convoluée à 8' - Masque convolué à 8'     #
                                #     Graphique 3 : Différence des graphiques 1 & 2                   #
                                #                                                                     # 
                                #######################################################################
                                
fig15, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

fig = plt.gcf()
fig.set_size_inches(16, 9)

test = ax1.imshow(convolution_locale)
ax1.set_xlabel("X (arcmin)")
ax1.set_ylabel("Y (arcmin)")
ax1.set_title(" Heatmap convolue 2' ")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig15.colorbar(test,cax=cax1)

test2 = ax2.imshow(convolution_grande)
ax2.set_xlabel("X (arcmin)")
ax2.set_ylabel("Y (arcmin)")
ax2.set_title(" Heatmap convolue 16' ")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig15.colorbar(test2,cax=cax2)

test3 = ax3.imshow(convolution_mask_locale)
ax3.set_xlabel("X (arcmin)")
ax3.set_ylabel("Y (arcmin)")
ax3.set_title(" Masque convolue 2' ")
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig15.colorbar(test3,cax=cax3)

test4 = ax4.imshow(convolution_mask_grande)
ax4.set_xlabel("X (arcmin)")
ax4.set_ylabel("Y (arcmin)")
ax4.set_title(" Masque convolue 16' ")
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="5%", pad=0.05)
fig15.colorbar(test4,cax=cax4)

fig15.tight_layout()    # Création d'espace pour les labels entre les subplots

fig15.savefig(outname15)


fig10, (ax1, ax2, ax3) = plt.subplots(1,3)

fig = plt.gcf()
fig.set_size_inches(16, 9)

mask = binary_mask == 0

A = np.ma.masked_array(convolution_locale, mask = mask)
B = np.ma.masked_array(convolution_mask_locale, mask = mask)

C = np.ma.masked_array(convolution_grande, mask = mask)
D = np.ma.masked_array(convolution_mask_grande, mask = mask)

#A_condition1 = np.where(A < 10**(-5),0,A)
#B_condition1 = np.where(B < 10**(-5),0,B)

step1 = A/B
step1 = np.ma.masked_array(step1, mask=mask)
step2 = C/D
step2 = np.ma.masked_array(step2, mask=mask)
S_N_map = step1 - step2



#step1_condition1 = np.where(step1 > 5, 0, step1) # Remplace toutes les valeurs > 100 par 0


fig_step1 = ax1.imshow(step1, interpolation='nearest')
ax1.set_xlabel("X (arcmin)")
ax1.set_ylabel("Y (arcmin)")
ax1.set_title("conv(data,2')/conv(completude,2')")
ax1.invert_yaxis()
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig10.colorbar(fig_step1,cax=cax1)

fig_step2 = ax2.imshow(step2, interpolation='nearest')
ax2.set_xlabel("X (arcmin)")
ax2.set_ylabel("Y (arcmin)")
ax2.set_title("conv(data,16')/conv(completude,16')")
ax2.invert_yaxis()
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig10.colorbar(fig_step2,cax=cax2)

# SUBSTRACT BOTH RESULTS

fig_S_N_map = ax3.imshow(S_N_map, interpolation='nearest')
ax3.set_xlabel("X (arcmin)")
ax3.set_ylabel("Y (arcmin)")
ax3.set_title("Difference des 2")
ax3.invert_yaxis()
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig10.colorbar(fig_S_N_map,cax=cax3)

fig10.tight_layout()    # Création d'espace pour les labels entre les subplots

fig10.savefig(outname10)



                                ###################################################################################################
                                #                                                                                                 #
                                #              Détermination de l'histogramme de la carte S/N                                     #
                                #              Détermination de la Gaussienne associée à l'histogramme                            #
                                #              Détermination de la valeur moyenne et de sigma à partir de la Gaussienne           #
                                #              Construction de la nouvelle carte S/N en terme de sigma                            #
                                #                                                                                                 #
                                ###################################################################################################
                                
new_SN_map_temp = S_N_map.flatten()       
new_SN_map = new_SN_map_temp[new_SN_map_temp != 0]                   

fig11 = plt.figure(11)
hist_SN_map = plt.hist(new_SN_map, bins = 500, normed = False)   
plt.xlabel("Number of stars per pixel")
plt.ylabel("Number of pixels ")
plt.title("Histogram S/N map")


		##########################################
		# Gaussienne ajustée à l'histogramme S/N #
		##########################################	
       

       
plt.xlim(min(new_SN_map), max(new_SN_map))			# Détermination des composantes de la Gaussienne

mean_SN = np.mean(new_SN_map)
variance_SN = np.var(new_SN_map)
sigma_SN = np.sqrt(variance_SN)

x_SN = np.linspace(min(new_SN_map), max(new_SN_map))

dx_SN = hist_SN_map[1][1] - hist_SN_map[1][0]
scale_SN = len(new_SN_map)*dx_SN
Gaussian_SN =  mlab.normpdf(x_SN,mean_SN,sigma_SN)*scale_SN
fig_Gaussian_SN = plt.plot(x_SN,Gaussian_SN)	# Traçage de la Gaussienne ajustée à la distribution (permet de ne pas normaliser l'histogramme)

fig11 = plt.savefig(outname11)

print " "
print "Pour une Gaussienne de 2' :"
print "La valeur de la moyenne est : " + str(mean_SN)
print "La valeur de la variance est : " + str(variance_SN)
print "La valeur de sigma est : " + str(sigma_SN)
print " "

        ############################################
        # Création de la carte finale S/N en sigma #
        ############################################

SN_map_final = (S_N_map - mean_SN) / sigma_SN
title = 'Signal to Noise map : ' + fieldname
print title

fig12 = plt.figure(12)
fig_SN_final = plt.imshow(SN_map_final)
cbar = plt.colorbar()
cbar.set_label(r'Signal to Noise (significance $\sigma$)')
plt.xlabel('X (arcmin)')
plt.ylabel('Y (arcmin)')
plt.title(title)
plt.gca().invert_yaxis()

fig12 = plt.savefig(outname12)

print 'Passage à la partie suivante : isochrone'

######################################################
######################################################


mask_G = np.bitwise_and( tbdata['G'] < 99.99, tbdata['GERR'] < 0.2, tbdata['GERR'] > 0.02)
mask_R = np.bitwise_and( tbdata['R'] < 99.99, tbdata['RERR'] < 0.2, tbdata['RERR'] > 0.02)


G_corrected = tbdata[mask_G]
R_corrected = tbdata[mask_R]

# Prmeière courbe
params_G = np.polyfit(G_corrected['G'], np.log(G_corrected['GERR']),1)
a = params_G[0]
A = np.exp(params_G[1])

# Deuxième courbe

params_R = np.polyfit(R_corrected['R'], np.log(R_corrected['RERR']),1)
b = params_R[0]
B = np.exp(params_R[1])

fig13 = plt.gcf()
fig13.set_size_inches(16, 9)


fig13, (ax1,ax2) = plt.subplots(1,2)

fig_error_g = ax1.plot(G_corrected['G'], (G_corrected['GERR']), '.')
g = ax1.plot(G_corrected['G'], (A*np.exp(a*G_corrected['G'])),'.')
ax1.set_xlabel('G')
ax1.set_ylabel('GERR')
ax1.set_ylim(0,0.3)
ax1.set_title('Evolution de GERR en fonction de G')


fig_error_r = ax2.plot((R_corrected['R']), (R_corrected['RERR']), '.')
fig_error_r = ax2.plot(R_corrected['R'], (B*np.exp(b*R_corrected['R'])),'.')
ax2.set_xlabel('R')
ax2.set_ylabel('RERR')
ax2.set_title('Evolution de RERR en fonction de R')

fig13.tight_layout() 

plt.savefig('graphique.png')

plt.show()'''
