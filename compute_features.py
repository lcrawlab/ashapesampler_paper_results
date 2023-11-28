#
# This code is based on the bioRxiv preprint "A Methodology for Morphological Feature Extraction and Unsupervised Cell Classification"
# authored by Bhaskar, D., Lee, D., Knútsdóttir, H., Tan, C., Zhang, M., Dean, P., Roskelley, C., Keshet, L.
# Source code on GitHub at: https://github.com/dbhaskar92/Research-Scripts/tree/master
#

from __future__ import division

import os
import gc
import sys
import csv
import math
import glob
import string
import warnings
import scipy.io
import extractcellfeatures

import numpy as NP
import pandas as PD

import matplotlib
import matplotlib.pyplot as PLT
import matplotlib.patches as patches

from matplotlib.path import Path
from matplotlib.patches import Ellipse, Polygon

from PIL import Image

from scipy import interpolate
from scipy.io import loadmat
from scipy.special import expit
from scipy import ndimage as NDI
from skimage.morphology import medial_axis, skeletonize, thin

matfilepaths = [y for x in os.walk("./") for y in glob.glob(os.path.join(x[0], '*.mat'))]
numcells = NP.shape(matfilepaths)[0]
print("Total number of segmented cells: " + repr(numcells))

outFolder = 'cell_feats_test'
if not os.path.isdir(outFolder):
    os.mkdir(outFolder)

cID = 0
px2um = 0.16
featureDict = dict()

for mat_file_path in matfilepaths:
    
    file_path_parts = mat_file_path.split('/')
    
    stimulant = file_path_parts[1]
    kPa_string = file_path_parts[2]
    kPa_string_parts = kPa_string.split('_')
    stiffness = int(kPa_string_parts[0])
    
    cellID_string = file_path_parts[3]
    cellID_string = cellID_string.split('_')[1]
    cellID_file = int(cellID_string.split('.')[0])
    
    mat_data = scipy.io.loadmat(mat_file_path)
    bin_img = mat_data['bin_mask']
    
    [labeled_img, num_labels] = NDI.measurements.label(bin_img)
    
    exp_metadata = [stimulant, stiffness, cellID_file, mat_file_path]
    
    if num_labels != 1:
        print("WARNING: cID: " + repr(cID) + " contains " + repr(num_labels) + " labelled objects.")
        
    # Crop
    bin_cell_mask = bin_img[NP.ix_(bin_img.any(1), bin_img.any(0))]
    (w, h) = NP.shape(bin_cell_mask)
    bin_mask = NP.zeros((w+50,h+50))
    bin_mask[25:w+25,25:h+25] = bin_cell_mask
    
    # Compute the medial axis (skeleton) and the distance transform
    skel, dist_transform = medial_axis(bin_mask, return_distance=True)
    morph_thinned = thin(bin_mask)

    # Distance to the background for pixels of the skeleton
    dist_on_skel = dist_transform * morph_thinned
    
    outFolder = 'cell_feats_test' + os.sep + 'cID_' + repr(cID)
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
        
    PIF = []
    X_px, Y_px = NP.where(bin_img == 1)
    for (lat_x, lat_y) in zip(X_px, Y_px):
        PIF.append([lat_x, lat_y])
        
    try:
        extractor = extractcellfeatures.ExtractFeatures(PIF)
        extractor.basic_props(3)
        extractor.shape_props()
        extractor.cell_centre_fit()
        extractor.moments()
        
    except ValueError:
        
        print("WARNING: skipped " + repr(cID))
        gc.collect()
        continue
        
    # Features
        
    cell_id = cID
    featureDict[cell_id] = [extractor.area_cell, extractor.perim_1sqrt2, extractor.equiv_diameter]
    featureDict[cell_id] = featureDict[cell_id] + [extractor.perim_3pv]
    featureDict[cell_id] = featureDict[cell_id] + [extractor.perim_poly, extractor.area_poly]
    featureDict[cell_id] = featureDict[cell_id] + extractor.ellipse_fvector + extractor.ccm_fvector
    featureDict[cell_id] = featureDict[cell_id] + [extractor.perim_eroded,  extractor.area_eroded]
    featureDict[cell_id] = featureDict[cell_id] + extractor.bdy_fvector
    featureDict[cell_id] = featureDict[cell_id] + extractor.shape_fvector
    featureDict[cell_id] = featureDict[cell_id] + extractor.ret_fvector
    featureDict[cell_id] = featureDict[cell_id] + extractor.hu_moments
    featureDict[cell_id] = featureDict[cell_id] + exp_metadata
    
    # Plotting
    
    fixed_perim = NP.transpose(extractor.perim_img)

    perim_img_ind = NP.where(fixed_perim == 1)

    xlim_min = min(perim_img_ind[1])
    xlim_max = max(perim_img_ind[1])

    ylim_min = min(perim_img_ind[0])
    ylim_max = max(perim_img_ind[0])

    U = extractor.spl_u
    OUT = interpolate.splev(U, extractor.spl_poly)
    BDY_FEATS = extractor.bdy_fvector
    
    ## Create circle, ellipse and polygon fit plot
    
    fig = PLT.figure(1)
    ax = fig.add_subplot(111, aspect='equal')
    
    c = PLT.Circle((extractor.ccm_fvector[0],
            extractor.ccm_fvector[1]),
            extractor.ccm_fvector[2])

    xlim_min = min(xlim_min, extractor.ccm_fvector[0] - extractor.ccm_fvector[2])
    xlim_max = max(xlim_max, extractor.ccm_fvector[0] + extractor.ccm_fvector[2])
    ylim_min = min(ylim_min, extractor.ccm_fvector[1] - extractor.ccm_fvector[2])
    ylim_max = max(ylim_max, extractor.ccm_fvector[1] + extractor.ccm_fvector[2])

    e = Ellipse(xy=NP.array([extractor.ellipse_fvector[0], extractor.ellipse_fvector[1]]),
            width = extractor.ellipse_fvector[4],
            height = extractor.ellipse_fvector[3],
            angle = extractor.ellipse_fvector[5]/(2*NP.pi)*360)
    
    a = extractor.ellipse_fvector[4]/2.0
    b = extractor.ellipse_fvector[3]/2.0
    alpha = extractor.ellipse_fvector[5]
    x_c = extractor.ellipse_fvector[0]
    y_c = extractor.ellipse_fvector[1]
    x_b = 0
    y_b = 0
    if a > b:
        x_b = abs(a * math.cos(alpha))
        y_b = abs(a * math.cos(alpha))
    else:
        x_b = abs(b * math.sin(alpha))
        y_b = abs(b * math.cos(alpha))

    xlim_min = min(xlim_min, x_c - x_b)
    xlim_max = max(xlim_max, x_c + x_b)
    ylim_min = min(ylim_min, y_c - y_b)
    ylim_max = max(ylim_max, y_c + y_b)

    PLT.imshow(fixed_perim, interpolation='nearest', cmap='Greys')
    perimeter = extractor.perim_3pv * px2um
    PLT.xticks([])
    PLT.yticks([])
    PLT.plot(extractor.perim_coord_poly[:,0], extractor.perim_coord_poly[:,1], 
             label='Polygon Fit (Perimeter = %.2f um)'%perimeter, color='g', lw=2)

    ax.add_artist(c)
    c.set_alpha(1)
    c.set_facecolor('none')
    c.set_edgecolor('blue')
    c.set_linewidth(3)
    c.set_label('Circle')

    ax.add_artist(e)
    e.set_alpha(1)
    e.set_facecolor('none')
    e.set_edgecolor('orange')
    e.set_linewidth(3)
    e.set_label('Ellipse')
    
    PLT.plot(0, 0, color='blue', label='Circle Fit (Variance = %.2f)'%extractor.ccm_fvector[5], lw=2)
    PLT.plot(0, 0, color='orange', label='Ellipse Fit (Variance = %.2f)'%extractor.ellipse_fvector[8], lw=2)
    
    lgd = PLT.legend(bbox_to_anchor=(0.0, 1.1, 1.0, 1.5), loc=3, ncol=1, mode=None, fontsize="small", borderaxespad=0.2, fancybox=True, shadow=True)

    padding = 10

    ax.set_xlim([xlim_min - padding, xlim_max + padding])
    ax.set_ylim([ylim_min - padding, ylim_max + padding])

    PLT.xticks([])
    PLT.yticks([])
    
    PLT.savefig(outFolder + os.sep + 'Fits.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    fig.clf()
    PLT.close()
    
    ## Create spline plot with boundary color based on magnitude and parity of curvature
    
    fig, (ax1, ax2) = PLT.subplots(1, 2, gridspec_kw = {'width_ratios':[1,10]})
    knorm = expit(extractor.spl_k/max(abs(extractor.spl_k))*10)

    norm = matplotlib.colors.Normalize(vmin=NP.min(extractor.spl_k), vmax=NP.max(extractor.spl_k))
    cb = matplotlib.colorbar.ColorbarBase(ax1, cmap=matplotlib.cm.coolwarm, norm=norm, orientation='vertical')
    cb.set_label('Magnitude of Curvature', labelpad=-100)

    pcolor = PLT.cm.coolwarm(knorm)

    for i in range(len(U)):
        ax2.plot(OUT[0][i:i+2], OUT[1][i:i+2], color=pcolor[i], linewidth=2)

    xlim_min = min(perim_img_ind[1])
    xlim_max = max(perim_img_ind[1])
    ylim_min = min(perim_img_ind[0])
    ylim_max = max(perim_img_ind[0])

    PLT.xticks([])
    PLT.yticks([])
    ax2.set_aspect(1)
    ax2.set_xlim([xlim_min - padding, xlim_max + padding])
    ax2.set_ylim([ylim_min - padding, ylim_max + padding])

    k_protrusions = BDY_FEATS[2]
    k_indentations = BDY_FEATS[3]
    PLT.gcf().text(0.01, 1.04, "Number of Protrusions: %d"%k_protrusions, fontsize=11)
    PLT.gcf().text(0.01, 0.98, "Number of Indentations: %d"%k_indentations, fontsize=11)
    
    PLT.savefig(outFolder + os.sep + 'SplineCurvature.png', bbox_inches='tight', dpi=200)
    fig.clf()
    PLT.close()
    
    ## Plot curvature function
    
    min_idx = NP.argmin(NP.absolute(extractor.spl_k))
    spl_k_shifted = NP.roll(extractor.spl_k, min_idx)
    x_data = range(0, len(extractor.spl_k))
    tick_data = NP.roll(x_data, min_idx) 

    fig, (ax1, ax2) = PLT.subplots(1, 2, figsize=(10,4), gridspec_kw = {'width_ratios':[8,2]})

    ax1.set_xticks(x_data[0:-1:50])
    ax1.set_xticklabels(tick_data[0:-1:50])
    ax1.plot(x_data, spl_k_shifted)
    ax1.set_xlabel('Spline Parameterization Index', fontsize=14)
    ax1.set_ylabel('Curvature', fontsize=14)

    ax2.boxplot(extractor.spl_k)
    ax2.set_xticks([])
    
    PLT.savefig(outFolder + os.sep + 'CurvatureFcn.png', bbox_inches='tight', dpi=200)
    fig.clf()
    PLT.close()
    
    ## Create spline plot with a binary boundary color scheme based on parity of curvature
    
    fig = PLT.figure(4)
    ax = fig.add_subplot(111, aspect='equal')
    
    knorm = NP.sign(extractor.spl_k)/2 + 0.5
    pcolor = PLT.cm.bwr(knorm)

    PLT.imshow(fixed_perim, interpolation='nearest', cmap='Greys')

    for i in range(len(U)):
        PLT.xticks([])
        PLT.yticks([])
        PLT.plot(OUT[0][i:i+2], OUT[1][i:i+2], color=pcolor[i], linewidth=2)

    ax.set_xlim([xlim_min - padding, xlim_max + padding])
    ax.set_ylim([ylim_min - padding, ylim_max + padding])

    PLT.xticks([])
    PLT.yticks([])

    PLT.plot(0, 0, color='blue', label='Negative Curvature', lw=2)
    PLT.plot(0, 0, color='red', label='Positive Curvature', lw=2)

    lgd = PLT.legend(bbox_to_anchor=(0.0, 1.1, 1.0, 1.5), loc=3, ncol=2, mode=None, fontsize="medium", 
                     borderaxespad=0.2, fancybox=False, shadow=False, frameon=False)
    
    PLT.savefig(outFolder + os.sep + 'SplineCurvatureBin.png', bbox_extra_artists=(lgd,), 
                bbox_inches='tight', dpi=200)
    fig.clf()
    PLT.close()
    
    ## Create oriented bounding rectangle plot
    
    fig = PLT.figure(5)
    ax = fig.add_subplot(111, aspect='equal')
    PLT.imshow(fixed_perim, interpolation='nearest', cmap='Greys')
    verts = extractor.ret_pvector
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
    path = Path(verts,codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='red', lw=2)
    ax.add_patch(patch)
    length = extractor.ferret_max * px2um
    width = extractor.ferret_min * px2um
    PLT.xticks([])
    PLT.yticks([])
    PLT.plot(0, 0, color='red', 
             label='Rectangle Fit (Length = %.2f um, Width = %.2f um)' % (length, width), lw=2)

    xlim_rec_min = min(verts[0][0], verts[3][0], verts[1][0], verts[2][0]) - padding
    xlim_rec_max = max(verts[1][0], verts[2][0], verts[0][0], verts[3][0]) + padding 
    ylim_rec_min = min(verts[3][1], verts[2][1], verts[0][1], verts[1][1]) - padding 
    ylim_rec_max = max(verts[0][1], verts[1][1], verts[3][1], verts[2][1]) + padding

    ax.set_xlim([xlim_rec_min, xlim_rec_max])
    ax.set_ylim([ylim_rec_min, ylim_rec_max])

    lgd = PLT.legend(bbox_to_anchor=(0.0, 1.1, 1.0, 1.5), loc=3, ncol=1, mode=None, 
                     fontsize="large", borderaxespad=0.2, fancybox=False, shadow=False, frameon=False)
    
    PLT.savefig(outFolder + os.sep + 'RectangularFit.png', bbox_extra_artists=(lgd,), 
                bbox_inches='tight', dpi=200)
    fig.clf()
    PLT.close()
    
    ## Medial axis
    
    fig = PLT.figure(6)
    PLT.imshow(dist_on_skel, cmap=PLT.cm.Spectral, interpolation='nearest')
    PLT.colorbar()
    PLT.contour(bin_mask, [0.5], colors='w')
    PLT.axis("off")
    PLT.savefig(outFolder + os.sep + 'MedialAxis.png', bbox_inches='tight', dpi=200)
    fig.clf()
    PLT.close()
    
    ## Save TDA feature
    NP.save(outFolder + os.sep + 'MedialAxis.npy', dist_on_skel)
    
    gc.collect()
    cID += 1

numCells = cID - 1
px2um = 0.16

feat_df = PD.DataFrame(columns = ['cid', 'area', 'perim_1sqrt2', 'equiv_diameter', 'perimeter', 
                                  'poly_perim', 'poly_area', 'ellipse_centroid_x', 'ellipse_centroid_y',
                                  'ellipse_eccentricity', 'ellipse_major_ax', 'ellipse_minor_ax',
                                  'ellipse_orient', 'ellipse_area', 'ellipse_perim', 'ellipse_var',
                                  'circle_centroid_x', 'circle_centroid_y', 'circle_radius', 'circle_perim',
                                  'circle_area', 'circle_var', 'eroded_perim', 'eroded_area', 
                                  'bdy_mean_curvature', 'bdy_std_curvature', 'bdy_num_protrusions', 
                                  'bdy_num_indentations', 'bdy_max_curvature', 'bdy_min_curvature', 
                                  'shape_extent', 'shape_euler', 'shape_solidity', 'shape_compactness', 
                                  'shape_elongation', 'shape_convexity', 'shape_circularity',
                                  'rect_centroid_x', 'rect_centroid_y', 'rect_orient', 
                                  'rect_ferret_max', 'rect_ferret_min',
                                  'Hu_1', 'Hu_2', 'Hu_3', 'Hu_4', 'Hu_5', 'Hu_6', 'Hu_7',
                                  'stimulant', 'stiffness', 'CellID_EXP', 'fullfilepath'])

for cell_id in range(1, numCells + 1):
    
    feat_df = feat_df.append({'cid': cell_id,
                              'area': featureDict[cell_id][0] * px2um * px2um,
                              'perim_1sqrt2': featureDict[cell_id][1] * px2um,
                              'equiv_diameter': featureDict[cell_id][2] * px2um,
                              'perimeter': featureDict[cell_id][3] * px2um,
                              'poly_perim': featureDict[cell_id][4] * px2um,
                              'poly_area': featureDict[cell_id][5] * px2um * px2um,
                              'ellipse_centroid_x': featureDict[cell_id][6],
                              'ellipse_centroid_y': featureDict[cell_id][7],
                              'ellipse_eccentricity': featureDict[cell_id][8],
                              'ellipse_major_ax': featureDict[cell_id][9] * px2um,
                              'ellipse_minor_ax': featureDict[cell_id][10] * px2um,
                              'ellipse_orient': featureDict[cell_id][11],
                              'ellipse_area': featureDict[cell_id][12] * px2um * px2um,
                              'ellipse_perim': featureDict[cell_id][13] * px2um,
                              'ellipse_var': featureDict[cell_id][14],
                              'circle_centroid_x': featureDict[cell_id][15],
                              'circle_centroid_y': featureDict[cell_id][16],
                              'circle_radius': featureDict[cell_id][17] * px2um,
                              'circle_perim': featureDict[cell_id][18] * px2um,
                              'circle_area': featureDict[cell_id][19] * px2um * px2um,
                              'circle_var': featureDict[cell_id][20],
                              'eroded_perim': featureDict[cell_id][21] * px2um,
                              'eroded_area': featureDict[cell_id][22] * px2um * px2um,
                              'bdy_mean_curvature': featureDict[cell_id][23],
                              'bdy_std_curvature': featureDict[cell_id][24],
                              'bdy_num_protrusions': featureDict[cell_id][25],
                              'bdy_num_indentations': featureDict[cell_id][26],
                              'bdy_max_curvature': featureDict[cell_id][27],
                              'bdy_min_curvature': featureDict[cell_id][28],
                              'shape_extent': featureDict[cell_id][29],
                              'shape_euler': featureDict[cell_id][30],
                              'shape_solidity': featureDict[cell_id][31],
                              'shape_compactness': featureDict[cell_id][32],
                              'shape_elongation': featureDict[cell_id][33],
                              'shape_convexity': featureDict[cell_id][34],
                              'shape_circularity': featureDict[cell_id][35],
                              'rect_centroid_x': featureDict[cell_id][36],
                              'rect_centroid_y': featureDict[cell_id][37],
                              'rect_orient': featureDict[cell_id][38],
                              'rect_ferret_max': featureDict[cell_id][39] * px2um,
                              'rect_ferret_min': featureDict[cell_id][40] * px2um,
                              'Hu_1': featureDict[cell_id][41],
                              'Hu_2': featureDict[cell_id][42],
                              'Hu_3': featureDict[cell_id][43],
                              'Hu_4': featureDict[cell_id][44],
                              'Hu_5': featureDict[cell_id][45],
                              'Hu_6': featureDict[cell_id][46],
                              'Hu_7': featureDict[cell_id][47],
                              'stimulant': featureDict[cell_id][48],
                              'stiffness': featureDict[cell_id][49],
                              'CellID_EXP': featureDict[cell_id][50],
                              'fullfilepath': featureDict[cell_id][51]}, ignore_index=True)
    

feat_df.to_csv('neutrophil_feats.csv', index=False)
