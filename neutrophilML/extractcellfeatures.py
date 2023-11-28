#
# This code is based on the bioRxiv preprint "A Methodology for Morphological Feature Extraction and Unsupervised Cell Classification"
# authored by Bhaskar, D., Lee, D., Knútsdóttir, H., Tan, C., Zhang, M., Dean, P., Roskelley, C., Keshet, L.
# Source code on GitHub at: https://github.com/dbhaskar92/Research-Scripts/tree/master
#
# Last modified: 21 Mar 2018
# Authors: Darrick Lee <y.l.darrick@gmail.com>, Dhananjay Bhaskar <dbhaskar92@gmail.com>, MoHan Zhang <mohan_z2hotmail.com>
# Description: Compute features from list of pixels representing a Cellular Potts Model (CPM) cell
#
from __future__ import division

import numpy as NP
import scipy.special
import skimage.measure
import skimage.morphology
import perimeter_3pvm
import matplotlib.pyplot as PLT

from operator import itemgetter
from scipy import interpolate
from scipy import ndimage as NDI
from detect_peaks import detect_peaks
from scipy.signal import argrelextrema

class ExtractFeatures:

    def __init__(self, cell_pixel_list):

        self.pix_list = cell_pixel_list

        self.cell_img = None # Binary image of full cell
        self.perim_img = None # Binary image of cell perimeter
        self.eroded_img = None # Binary image of eroded cell perimeter

        self.perim_coord = None # Coordinates of the cell perimeter pixels
        self.perim_coord_dp = None # Coordinates of approximate perimeter (Douglas-Peucker)
        self.perim_coord_poly = None # Coordinates of polygon (derived from 3pv)
        self.perim_coord_eroded = None # Coordinates of eroded polygon

        self.area_cell = None # Area of cell by pixel counting
        self.area_poly = None # Area of polygon derived from 3pv
        self.area_eroded = None # Area of eroded polygon

        self.perim_3pv = None # Perimeter of cell using 3pv method
        self.perim_1sqrt2 = None # 1 sqrt 2 method
        self.perim_poly = None # Perimeter of polygon derived from 3pv
        self.perim_eroded = None # Perimeter of polygon of eroded cell

        self.equiv_diameter = None # The equivalent diameter of a circle with same area as cell
        self.shape_factor = None

        self.ellipse_fvector = None # Features from ellipse fit
        self.shape_fvector = None # Shape factors
        self.ccm_fvector = None # Features from CCM fit
        self.ret_pvector = None # Rectangular fit, for plotting
        self.ret_fvector = None # Rectangular fit, feature vector
        self.bdy_fvector = None # Features based on boundary

        self.spl_poly = None # Spline tck variables approximating 3pv-polygon
        self.spl_u = None # Spline parameter
        self.spl_k = None # Spline curvature

        self.spl_min_idx = None # Index of curvature spline closest to zero
        self.protrusion_indices = None # Indices of curvature maxima
        self.indentation_indices = None # Indices of curvature minima
        
        self.ferret_max = None # Maximum ferret diameter for the cell
        self.ferret_min = None # Minimum ferret diameter for the cell

        self.hu_moments = None # Array of 7 Hu-moments

        self.cell_to_image()

        # Check how many connected components there are
        s = [[1,1,1],[1,1,1],[1,1,1]] # Used to allow diagonal connections
        _, self.connectedComp = NDI.measurements.label(self.cell_img,structure=s)

    def cell_to_image(self):

        # Find x, y coordinate bounds
        x_res = max(self.pix_list, key=itemgetter(0))[0]
        y_res = max(self.pix_list, key=itemgetter(1))[1]

        # Creating labeled_img
        self.cell_img = NP.zeros([x_res+2, y_res+2], dtype=NP.int_)

        for (x_pix, y_pix) in self.pix_list:
            self.cell_img[x_pix-1, y_pix-1] = 1

        # Find the pixels that make up the perimeter
        eroded_image = NDI.binary_erosion(self.cell_img)

        eroded_image_open = NDI.binary_opening(eroded_image, structure=NP.ones((3,3)))
        eroded_image_open2 = NDI.binary_erosion(eroded_image_open)
        
        self.eroded_img = NP.bitwise_xor(eroded_image_open, eroded_image_open2)
        self.perim_img = NP.bitwise_xor(self.cell_img, eroded_image)

        # Create a list of the coordinates of the pixels (use the center of the pixels)
        perim_image_ind = NP.where(self.perim_img == 1)
        perim_image_coord = NP.array([perim_image_ind[0], perim_image_ind[1]])
        self.perim_coord = NP.transpose(perim_image_coord)

        return


    def basic_props(self, splineSmooth=10):

        '''
        Description: Calculates the perimeter and area using basic methods. For perimeter,
        we use the 3pv, 1 sqrt2 method, and look at the 3pv-polygon perimeter. For area,
        we use pixel counting, and look at the 3pv-polygon area.

        For 3pv perimeter: Use three-pixel vector method to compute perimeter and shape factor
        Reference: http://www.sciencedirect.com/science/article/pii/0308912687902458

        For cubic spline: Use the built-in function from scipy. Note about smoothing parameter in reference.
        Reference: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        '''

        # Perimeter: 3pv and polygon perimeter (polygon from 3pv)
        self.perim_3pv, self.perim_poly, self.perim_coord_poly = perimeter_3pvm.perimeter_3pvm(self.perim_img)
        _, self.perim_eroded, self.perim_coord_eroded = perimeter_3pvm.perimeter_3pvm(self.eroded_img)

        # Perimeter: Approximate polygon using Douglas-Peucker algorithm
        self.perim_coord_dp = skimage.measure.approximate_polygon(NP.array(self.perim_coord_poly), 0.75)

        # Create cubic spline
        self.spl_poly, _ = interpolate.splprep(NP.transpose(self.perim_coord_poly), per=1) # s=splineSmooth
        self.spl_u = NP.linspace(0, 1.0, 500)

        # Calculate spline curvature
        D1 = interpolate.splev(self.spl_u, self.spl_poly, der=1)
        D2 = interpolate.splev(self.spl_u, self.spl_poly, der=2)
        self.spl_k = (D1[0]*D2[1] - D1[1]*D2[0])/((D1[0]**2+D1[1]**2)**(3./2))

        # Calculate boundary features
        mean_curvature = NP.mean(self.spl_k)
        std_dev_curvature = NP.std(self.spl_k)
        
        self.spl_min_idx = NP.argmin(NP.absolute(self.spl_k))
        spl_k_shifted = NP.roll(self.spl_k, self.spl_min_idx)
        
        self.protrusion_indices = detect_peaks(spl_k_shifted, mph=0.35, mpd=10)
        num_protrusions = len(self.protrusion_indices)
        
        self.indentation_indices = detect_peaks(spl_k_shifted, mph=0.35, mpd=10, valley=True)
        num_indentations = len(self.indentation_indices)
        
        global_max_curvature = max(self.spl_k)
        global_min_curvature = min(self.spl_k)

        self.bdy_fvector = [mean_curvature, std_dev_curvature, num_protrusions, num_indentations, 
                            global_max_curvature, global_min_curvature]

        # Perimeter: 1 sqrt2 (function from regionprops)
        props = skimage.measure.regionprops(self.cell_img)
        self.perim_1sqrt2 = props[0].perimeter

        # Area: Pixel Counting
        self.area_cell = len(self.pix_list)

        # Area: Polygon area (from 3pv polygon)
        # Extract x and y coordinates
        # We subtract 0.5 because PLT.imshow() shows coordinates as the centers of pixels
        # Using the shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
        YY = self.perim_coord_poly[:,0]
        XX = self.perim_coord_poly[:,1]

        self.area_poly = 0.5*NP.abs(NP.dot(XX,NP.roll(YY,1))-NP.dot(YY,NP.roll(XX,1)))

        # Area: Eroded polygon area
        YY = self.perim_coord_eroded[:,0]
        XX = self.perim_coord_eroded[:,1]

        self.area_eroded = 0.5*NP.abs(NP.dot(XX,NP.roll(YY,1))-NP.dot(YY,NP.roll(XX,1)))

        # Equivalent Diameter: The diameter of the circle with the same area (pixel counted) as the cell
        self.equiv_diameter = NP.sqrt(4*self.area_cell/NP.pi)

        return

    def cot(self,angle):
    
        return 1/NP.tan(angle)

    def rect(self):
    
        '''
                Calculate oriented bounding rectangle
                Paper: 
        '''
        
        # list of boundary points
        boundary_points = self.perim_coord

        # size of boundary points list
        X, Y = boundary_points.shape

        # compute centroid
        #x_sum = 0
        #y_sum = 0
        #for i in range(X):
        #    x = boundary_points[i][0]
        #    y = boundary_points[i][1]
        #    x_sum += x
        #    y_sum += y
        #y_bar = x_sum/X
        #x_bar = y_sum/X
        
        # Alternative centroid computation
        props = skimage.measure.regionprops(self.cell_img)
        centroid = props[0].centroid
        x_bar = centroid[0]
        y_bar = centroid[1]

        # compute angle using the formula tan(2*angle) = 2*(sum((xi-xbar)*(yi-ybar))/((xi-xbar)^2-(yi-ybar)^2))
        top_sum = 0
        bot_sum = 0
        for i in range(X):
            x = boundary_points[i][0]
            y = boundary_points[i][1]
            top_sum += (x-x_bar)*(y-y_bar) # sum((xi-xbar)*(yi-ybar)
            bot_sum += ((x-x_bar)**2-(y-y_bar)**2) # (xi-xbar)^2-(yi-ybar)^2)
        angle = (NP.arctan((2*top_sum)/bot_sum))/2
        major_upper = [] # list of boundary points upper of major axes
        major_lower = []
        minor_upper = []
        minor_lower = []

        # initialize max/min distances to the axes
        major_max = 0
        major_min = 0
        minor_max = 0
        minor_min = 0

        # initialize furthest point to the axes
        major_up_point = (0,0) # point on top of major axes that is the furthest away
        major_low_point = (0,0) # point below major axes that is the furthest away
        minor_up_point = (0,0) # point to the right of minor axes that is the furthest away
        minor_low_point = (0,0) # point to the left of minor axes that is the furthest away

        # classify each boundary point, calculate distance, and record max distances seen so far
        for i in range(X):
            x = boundary_points[i][0]
            y = boundary_points[i][1]
            V_major = (y-y_bar)-NP.tan(angle)*(x-x_bar)
            V_minor = (y-y_bar)+self.cot(angle)*(x-x_bar)
            dist_major = ((x-x_bar)*NP.sin(angle)-(y-y_bar)*NP.cos(angle))**2
            dist_minor = ((x-x_bar)*NP.cos(angle)+(y-y_bar)*NP.sin(angle))**2
            if (V_major > 0):
                major_upper.append(boundary_points[i])
                if (dist_major > major_max):
                    major_max = dist_major
                    major_up_point = boundary_points[i]
            elif (V_major < 0):
                major_lower.append(boundary_points[i])
                if (dist_major > major_min):
                    major_min = dist_major
                    major_low_point = boundary_points[i]
            if (V_minor > 0):
                minor_upper.append(boundary_points[i])
                if (dist_minor > minor_max):
                    minor_max = dist_minor
                    minor_up_point = boundary_points[i]
            elif (V_minor < 0):
                minor_lower.append(boundary_points[i])
                if (dist_minor > minor_min):
                    minor_min = dist_minor
                    minor_low_point = boundary_points[i]

        # calculate vertices of the rectangle based on points classified
        t_lx = (major_up_point[0]*NP.tan(angle)+minor_up_point[0]*self.cot(angle)+minor_up_point[1]-major_up_point[1])/(NP.tan(angle)+self.cot(angle))
        t_ly = (major_up_point[1]*self.cot(angle)+minor_up_point[1]*NP.tan(angle)+minor_up_point[0]-major_up_point[0])/(NP.tan(angle)+self.cot(angle))

        t_rx = (major_up_point[0]*NP.tan(angle)+minor_low_point[0]*self.cot(angle)+minor_low_point[1]-major_up_point[1])/(NP.tan(angle)+self.cot(angle))
        t_ry = (major_up_point[1]*self.cot(angle)+minor_low_point[1]*NP.tan(angle)+minor_low_point[0]-major_up_point[0])/(NP.tan(angle)+self.cot(angle))

        b_lx = (major_low_point[0]*NP.tan(angle)+minor_up_point[0]*self.cot(angle)+minor_up_point[1]-major_low_point[1])/(NP.tan(angle)+self.cot(angle))
        b_ly = (major_low_point[1]*self.cot(angle)+minor_up_point[1]*NP.tan(angle)+minor_up_point[0]-major_low_point[0])/(NP.tan(angle)+self.cot(angle))

        b_rx = (major_low_point[0]*NP.tan(angle)+minor_low_point[0]*self.cot(angle)+minor_low_point[1]-major_low_point[1])/(NP.tan(angle)+self.cot(angle))
        b_ry = (major_low_point[1]*self.cot(angle)+minor_low_point[1]*NP.tan(angle)+minor_low_point[0]-major_low_point[0])/(NP.tan(angle)+self.cot(angle))

        # make the vertices into tuples
        top_left = (t_lx,t_ly)
        top_right = (t_rx, t_ry)
        bot_left = (b_lx,b_ly)
        bot_right = (b_rx,b_ry)

        # return the list of vertices, top_left twice so that plt could join the sides of the rectangle
        self.ret_pvector=[top_left,top_right,bot_right,bot_left,top_left]
        
        # Calculate Ferret diameters (max and min) using vertices of the minimum bounding rectange)
        d1 = NP.sqrt((top_left[0]-top_right[0])**2 + (top_right[1]-top_left[1])**2)
        d2 = NP.sqrt((top_left[0]-bot_left[0])**2 + (bot_left[1]-top_left[1])**2)
        self.ferret_max = max(d1,d2)
        self.ferret_min = min(d1,d2)
        
        centroid = (x_bar, y_bar)
        self.ret_fvector = [centroid[0], centroid[1], angle, self.ferret_max, self.ferret_min]


    def shape_props(self):

        '''
        Description: Returns list of properties derived from fitting ellipse (in the following order)
        centroid_x, centroid_y, eccentricity, majorAxisLength, minorAxisLength, orientation,
        area and perimeter.

        This uses regionprops() fom skimage.measure. The ellipse fit is done by
        fitting an ellipse with the same second central moment as the image. By looking
        at the code, this is done by calculating the inertia tensor of the matrix,
        finding the eigenvalues (the second central moments using the principal axes),
        and matching those with the equations for second central moment of an ellipse.

        In addition, this method returns a set of shape factors such as extent, euler number,
        solidity, compactness, elongation, convexity, and circularity.

        NOTE: We assume that basic_props() has already been called.

        Reference: https://en.wikipedia.org/wiki/Image_moment

        Ellipse perimeter: Equation given in https://en.wikipedia.org/wiki/Ellipse#Circumference
        The elliptic integral of the second kind implemented in scipy:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html#scipy.special.ellipe
        Note that the scipy definition of the integral differs slightly than wiki, so we take E(e^2) rather than E(e).
        '''

        props = skimage.measure.regionprops(self.cell_img)

        centroid = props[0].centroid

        # Calculate ellipse variance
        perim_coord = NP.transpose(self.perim_coord)
        numPt = len(perim_coord[0])
        cov = NP.mat(NP.cov(perim_coord))
        V = NP.array([perim_coord[0] - centroid[0], perim_coord[1] - centroid[1]])
        cV = NP.array(NP.linalg.inv(cov)*NP.mat(V))
        d = NP.sqrt(V[0]*cV[0] + V[1]*cV[1])
        mu = NP.sum(d)/numPt
        sigma = NP.sqrt(NP.sum((d-mu)**2)/numPt)
        
        major_axis_len = props[0].major_axis_length
        minor_axis_len = props[0].minor_axis_length
        semi_major_len = major_axis_len/2.0
        semi_minor_len = minor_axis_len/2.0
        ecc = NP.sqrt(1-((semi_minor_len)**2/(semi_major_len**2)))
        ellipse_prop_list = [centroid[0]]
        ellipse_prop_list.append(centroid[1])
        ellipse_prop_list.append(ecc)
        ellipse_prop_list.append(major_axis_len)
        ellipse_prop_list.append(minor_axis_len)
        ellipse_prop_list.append(props[0].orientation) # In degrees starting from the x-axis
        ellipse_prop_list.append(NP.pi*semi_major_len*semi_minor_len) # Ellipse area
        ellipse_prop_list.append(4.0*semi_major_len*scipy.special.ellipe(ecc)) # Ellipse perimeter
        ellipse_prop_list.append(sigma/mu) # Ellipse variance

        self.ellipse_fvector = ellipse_prop_list

        # NOTE: For shape factors, we use the perim_poly for perimeter, and pixel counting for area

        # Calculate values needed for shape factors
        inertia_ev = props[0].inertia_tensor_eigvals
        area = self.area_cell
        perim = self.perim_poly

        # Calculate convex hull perimeter
        cvx_img = props[0].convex_image # Find the pixels that make up the perimeter
        eroded_cvx_img = NDI.binary_erosion(cvx_img)
        cvx_perim_img = NP.bitwise_xor(cvx_img, eroded_cvx_img)
        cvx_perim_img = NP.lib.pad(cvx_perim_img,(1,1),'constant') # Pad with 0's for perimeter code to work properly
        _, cvx_perim, _ = perimeter_3pvm.perimeter_3pvm(cvx_perim_img)

        # Calculate shape factors
        if (self.ret_fvector == None):
            self.rect()
        compactness = NP.sqrt((4*area)/NP.pi)/(self.ferret_max)
        elongation = 1-(self.ferret_min/self.ferret_max) 
        convexity = cvx_perim/perim
        circularity = 4*NP.pi*area/(perim**2)

        # Create shape feature vector
        self.shape_fvector = []
        self.shape_fvector.append(props[0].extent) # Ratio of pixels in the region to pixel in bounding box (from 0 to 1)
        self.shape_fvector.append(props[0].euler_number) # Euler number
        self.shape_fvector.append(props[0].solidity) # Ratio of pixels in the region to pixels of the convex hull image (from 0 to 1)
        self.shape_fvector.append(compactness) # sqrt(4(area)/pi)/(Max ferret diameter)
        self.shape_fvector.append(elongation) # # 1 - aspect ratio
        self.shape_fvector.append(convexity) # Ratio of convex hull perimeter to perimeter (from 0 to 1)
        self.shape_fvector.append(circularity) # Ratio of area to perimeter squared (circle = 1, starfish << 1)

        return


    def cell_centre_fit(self):

        '''
        Description: Returns a list of features derived from fitting a circle (in the following order):
        centroid_x, centroid_y, radius, perimeter, area.

        This uses a least-squares estimator for the circle, using the points on the boundary of the cell.
        These points are chosen to be at the center of the boundary pixels.

        Circle variance is a goodness of fit measure for the circle fit and is defined in this reference:
        http://www.math.uci.edu/icamp/summer/research_11/park/shape_descriptors_survey.pdf
        '''

        c_model = skimage.measure.CircleModel()
        c_model.estimate(self.perim_coord)

        if skimage.__version__ == '0.9.3':
            (xc, yc, r) = c_model._params
        else:									# For newer versions
            (xc, yc, r) = c_model.params
            
        perim_coord = NP.transpose(self.perim_coord)
        x = perim_coord[0]
        y = perim_coord[1]
        x_m = NP.mean(x)
        y_m = NP.mean(y)
        
        u = x - x_m
        v = y - y_m
        Suv = sum(u*v)
        Suu = sum(u**2)
        Svv = sum(v**2)
        Suuv = sum(u**2 * v)
        Suvv = sum(u * v**2)
        Suuu = sum(u**3)
        Svvv = sum(v**3)
        A = NP.array([ [ Suu, Suv ], [Suv, Svv]])
        B = NP.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
        uc, vc = NP.linalg.solve(A, B)
        xc = x_m + uc
        yc = y_m + vc
        Ri = NP.sqrt((x-xc)**2 + (y-yc)**2)
        r = NP.mean(Ri)

        # Calculate the circle variance
        numPt = len(perim_coord[0])
        d = NP.sqrt((perim_coord[0]-xc)**2 + (perim_coord[1]-yc)**2)
        mu = NP.sum(d)/numPt
        sigma = NP.sqrt(NP.sum((d-mu)**2)/numPt)

        cell_centre_features = [xc]
        cell_centre_features.append(yc)
        cell_centre_features.append(r)
        cell_centre_features.append(2*NP.pi*r)
        cell_centre_features.append(NP.pi*r**2)
        cell_centre_features.append(sigma/mu) # Circle variance (lower is better)

        self.ccm_fvector = cell_centre_features
        
        return

    def moments(self):

        '''
        Description: Returns the 7 Hu-moments, Legendre-moment, as well as Zernicke-moment

        '''
        
        props = skimage.measure.regionprops(self.cell_img)
        self.hu_moments = list(props[0].moments_hu)
        
        return
