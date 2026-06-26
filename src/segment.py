import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import scipy.ndimage as ndi
from scipy.ndimage import generic_filter
from scipy.optimize import minimize, leastsq
from scipy.spatial import ConvexHull
import skimage.segmentation as seg
import skimage.morphology as morph
import skimage.measure as measure
import skimage.draw as draw
from src.bwmorph import *
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger('tensionmap')

class VMSI_obj:
    def __init__(self):
        self.V_df = []
        self.C_df = []
        self.E_df = []

class Segmenter:
    def __init__(self, images = None, masks = None, very_far = 300, labelled=False):
        """
        :param: images: (Numpy array) Membrane-stained images to be segmented. WARNING: currently experimental and not working as intended. Default: None.
        :param masks: (Numpy array) Segmented image with edges set to zero and cells set to non-zero. Edges must be 1px wide and 4-connected. Default: None.
        :param very_far: (Int) Maximum distance in pixels between two vertices connected by the same edge. Default: 300.
        :param labelled: (Bool) Whether the segmented cells have been labelled. Default: False.
        """
        self.images = []
        self.masks = []
        self.very_far = very_far

        if images is not None:
            self.images = images
        if masks is not None:
            self.masks = masks
        if not labelled:
            self.masks = measure.label(self.masks)

    def process_segmented_image(self, holes_mask=None):
        """
        Given a segmented mask, produce VMSI_obj for input into VMSI
        """
        # Before processing mask, obtain polygon perimeter and original image label for each cell
        polygon_perimeter = self.polygon_perimeter()
        # Process mask

        # Clear border (create external cell from all cells that run into image boundary)
        tmp1 = seg.clear_border(self.masks)
        #print('tmp1 beginning', np.unique(tmp1))
        # If we are specifying holes, also set cells bordering holes as external cell
        if holes_mask is not None:
            tmp5 = morph.binary_dilation(holes_mask, footprint=np.ones([5,5]))
            hole_adj_cells = np.unique(tmp5 * self.masks)[1:]
            tmp6 = np.isin(self.masks, hole_adj_cells)
            tmp1[tmp6] = 0
        tmp2 = ((self.masks - tmp1)>0).astype(int)
        tmp3 = tmp1 + tmp2
        #print('tmp1 after holes', np.unique(tmp1))

        # Find edge pixels that only separate external cells.
        # Vectorized replacement for generic_filter(tmp3, len(set(neighborhood))):
        # A background pixel has < 3 unique values in its 3x3 window iff all non-zero
        # neighbors share the same label (or there are none at all).
        local_max = ndi.maximum_filter(tmp3, size=3)
        tmp3_min = np.where(tmp3 == 0, tmp3.max() + 1, tmp3)
        local_min_nz = ndi.minimum_filter(tmp3_min, size=3)
        few_unique = (local_max == 0) | (local_max == local_min_nz)
        tmp1[np.logical_and(tmp1==0, few_unique)] = 1
        #print('tmp1 after logical and', np.unique(tmp1))
        mask_tmp = tmp1
        #print('mask_tmp', np.unique(mask_tmp))

        # Relabel mask (may not be necessary in future, just for Matlab compatibility)
        mask_tmp = self.relabel(mask_tmp)

        # Create VMSI object to store vertex, cell and edge information
        obj = VMSI_obj()

        obj.C_df = self.find_cells(mask_tmp)
        # Add polygon perimeter and cell label information to C_df

        cell_pwdist = cdist(polygon_perimeter[['centroid_x','centroid_y']], np.array(obj.C_df['centroids'].tolist()))
        matching_cells = np.where(cell_pwdist<=2)
        obj.C_df.loc[obj.C_df.index.values[matching_cells[1]],['label','polygon_perimeter']] = polygon_perimeter[['label','polygon_perimeter']].values[matching_cells[0]]

        obj.V_df, cc = self.find_vertices(mask_tmp, obj.C_df)
        obj.E_df = self.find_edges(obj, mask_tmp, cc)
        self.identify_holes(obj, holes_mask)
        return obj, mask_tmp

    def find_vertices(self, mask, C_df):
        branchpoints = self.find_branch_points(mask==0)

        cc = measure.label(branchpoints, connectivity=2)
        v = np.array([np.flip(np.round(regionprops.centroid).astype(int)) for regionprops in measure.regionprops(cc)])
        a = np.array([regionprops.coords for regionprops in measure.regionprops(cc)])
        # regionprops returns the coordinates in numpy indexing rather than cartesian indexing - e.g.
        # (rows, cols) rather than (x, y) so flip and re-sort coordinates
        a = a[v[:,0].argsort()]
        v = v[v[:,0].argsort()]

        R = np.zeros([2,v.shape[0]])
        # Accumulate vertex rows in a list; do a single pd.DataFrame() call at the end
        vertex_rows = []
        for i in range(v.shape[0]):

            vertex = v[i,:]
            # Flip again to convert back to numpy indexing
            r0 = max(0, min(a[i][:,0])-1)
            r1 = min(mask.shape[0], max(a[i][:,0])+2)
            c0 = max(0, min(a[i][:,1])-1)
            c1 = min(mask.shape[1], max(a[i][:,1])+2)
            ncells = mask[r0:r1, c0:c1]
            ncells = np.unique(ncells[ncells!=0])-1

            vertex_rows.append({'coords': vertex, 'ncells': ncells,
                                 'nverts': np.array([]), 'edges': np.array([])})

            R[0, i] = vertex[0]
            R[1, i] = vertex[1]

        # Single concat — avoids O(n²) DataFrame copies inside the loop
        if vertex_rows:
            V_df = pd.DataFrame(vertex_rows)
        else:
            V_df = pd.DataFrame(columns=['coords', 'ncells', 'nverts', 'edges'])
        # Identify neighbour vertices
        adj = np.zeros([v.shape[0],v.shape[0]])

        D = np.add(np.tile(np.sum(np.multiply(R, R), axis=0), (v.shape[0],1)),
                   np.tile(np.sum(np.multiply(R, R), axis=0), (v.shape[0],1)).T) - 2*np.matmul(R.T, R)

        for V in range(len(V_df)):
            for cell in V_df.at[V, 'ncells']:
                C_df.at[cell, 'numv'] += 1
                C_df.at[cell, 'nverts'] = np.append(C_df.at[cell, 'nverts'], np.array([V]))

        for C in range(len(C_df)):
            # If cell has no vertices, assume it must border the external cell only
            if C_df.at[C, 'nverts'].size > 0:
                ncells = np.setdiff1d(np.unique(np.hstack(V_df.loc[C_df.at[C, 'nverts'], 'ncells'].tolist())), C)
            else:
                ncells = np.array([0])
            C_df.at[C, 'ncells'] = ncells

        for i in range(v.shape[0]):
            for j in range(i+1,v.shape[0]):
                if D[i,j] <= np.power(self.very_far, 2):
                    v1_ncells = V_df['ncells'].iloc[i]
                    v1_ncells = v1_ncells[v1_ncells != 0]
                    v2_ncells = V_df['ncells'].iloc[j]
                    v2_ncells = v2_ncells[v2_ncells != 0]

                    if np.intersect1d(v1_ncells, v2_ncells).size >=2:
                        adj[i,j] = 1
                        adj[j,i] = 1
            V_df['nverts'].iloc[i] = np.where(adj[i,:]==1)[0]
        return V_df, cc

    def find_branch_points(self, skel):
        # Vectorized branch point finding; faster than convolving with filter

        skel = np.array(skel, dtype=int)

        branch_points = np.zeros(skel.shape)
        branch_points[1:skel.shape[0]-1,1:skel.shape[1]-1] = skel[2:skel.shape[0],1:skel.shape[1]-1] + skel[0:skel.shape[0]-2,1:skel.shape[1]-1] + \
                                   skel[1:skel.shape[0]-1,2:skel.shape[1]] + skel[1:skel.shape[0]-1,0:skel.shape[1]-2]
        branch_points = np.multiply(branch_points,skel)
        branch_points = branch_points >= 3
        return branch_points

    def find_cells(self, mask):
        # Identify cells, record region information
        # regionprops returns the co-ordinates in numpy indexing rather than cartesian indexing - e.g.
        # (rows, cols) rather than (x, y) so flip

        # Single regionprops call — consolidates the previous 5 separate calls
        props = measure.regionprops(mask)
        cell_props = pd.DataFrame(measure.regionprops_table(
            mask, properties=('label', 'feret_diameter_max', 'area')))

        c = np.array([np.flip(p.centroid) for p in props])
        peri = np.array([p.perimeter for p in props])
        ine = np.array([p.inertia_tensor[np.triu_indices(2)] for p in props])
        bbox = np.array([[p.bbox[3]-p.bbox[1], p.bbox[2]-p.bbox[0]] for p in props])
        moments_hu = np.array([p.moments_hu for p in props])

        logger.debug('find_cells: mask=%s unique=%s', mask.shape, np.unique(mask))
        logger.debug('find_cells: %d props, peri.shape=%s, c.shape=%s', len(props), peri.shape, c.shape)
        # estimate very_far to be the half the maximum cell perimeter
        self.very_far = np.max(peri[1:]) / 2

        n = c.shape[0]
        # Build DataFrame in one shot instead of O(n²) pd.concat inside a loop
        C_df = pd.DataFrame({
            'centroids': list(c),
            'nverts': [np.array([]) for _ in range(n)],
            'numv': np.zeros(n, dtype=int),
            'ncells': [np.array([]) for _ in range(n)],
            'edges': [np.array([]) for _ in range(n)],
            'area': cell_props['area'].values,
            'holes': [False] * n,
            'inertia': list(ine),
            'perimeter': peri,
            'polygon_perimeter': np.zeros(n),
            'feret_d': cell_props['feret_diameter_max'].values,
            'moments_hu': list(moments_hu),
            'bbox': list(bbox),
            'label': np.zeros(n, dtype=int),
        })
        return C_df

    def identify_holes(self, obj, holes_mask):
        """
        Filter out labelled objects that have area greater than 2x the median area and are non-convex
        """
        areas = obj.C_df['area'].to_numpy()
        for i in range(obj.C_df.shape[0]):
            vcoords = np.array(obj.V_df.loc[obj.C_df.at[i, 'nverts'], 'coords'].tolist())
            centroid = np.array(obj.C_df.at[i, 'centroids']).astype(int)
            vcoords_unique_xs = len(set([p[0] for p in vcoords]))
            vcoords_unique_ys = len(set([p[1] for p in vcoords]))

            if vcoords.shape[0] >= 3 and (holes_mask is None or holes_mask[centroid[1], centroid[0]] == 0):
                hull = ConvexHull(vcoords)
                if hull.simplices.shape[0] < vcoords.shape[0] and obj.C_df.at[i, 'area'] > 3*np.median(areas):
                    obj.C_df.at[i, 'holes'] = True
            else:
                obj.C_df.at[i, 'holes'] = True
        return

    def relabel(self, mask):
        """
        If cells aren't sequenctially label, relabel them
        """
        ids = np.sort(np.unique(mask))
        # searchsorted maps each mask pixel value to its rank in ids (background 0 stays 0)
        return np.searchsorted(ids, mask).astype(int)

    def find_edges(self, obj, mask, cc):
        l_dat = mask
        b_dat = (l_dat == 0).astype(int)

        rv = np.vstack(obj.V_df['coords'])
        verts = np.zeros(b_dat.shape)
        verts[rv[:,1],rv[:,0]] = 1

        b_dat[np.where(verts == 1)] = 0
#        b_end  = b_dat * morph.dilation(verts, morph.disk(1))

        b_dat[cc != 0] = 0
#        b_end = (b_end * b_dat) + (self.endpoints(b_dat) * b_dat)
        # Not sure what the Matlab code is trying to accomplish but it doesn't seem to work so try another method
        b_end = self.endpoints(b_dat) * b_dat

        re = np.argwhere(b_end.T != 0)
        D = cdist(re, rv)

        b_l = measure.label(b_dat.T, connectivity=1).T
        end_labels = b_l[re[:,1],re[:,0]]
        b_props = measure.regionprops(b_l)

        # Accumulate edge rows; single pd.DataFrame() call replaces O(n²) pd.concat inside a loop
        edge_rows = []
        for i in range(1, len(np.unique(b_l))):
            end_points = np.argwhere(end_labels==i)

            v1 = -1
            v2 = -1

            # Edges with 1 endpoint are generally 1-length; ignore these
            if len(end_points) == 2:
                v1 = np.argmin(D[end_points[0],:])
                v2 = np.argmin(D[end_points[1],:])
                if (v1 == v2):
                    sort1 = np.sort(D[end_points[0],:]).squeeze()
                    sort2 = np.sort(D[end_points[1],:]).squeeze()
                    if abs(sort1[0] - sort1[1]) <= np.sqrt(3):
                        v1 = np.argsort(D[end_points[0],:]).squeeze()[1]
                    elif abs(sort2[0] - sort2[1]) <= np.sqrt(3):
                        v2 = np.argsort(D[end_points[1],:]).squeeze()[1]

            if (v1 != -1) and (v2 != -1) and (v2 in obj.V_df.at[v1, 'nverts']) and ((v1 not in obj.C_df.at[0, 'nverts']) or (v2 not in obj.C_df.at[0, 'nverts'])):
                pix = np.ravel_multi_index(np.flip(b_props[i-1].coords.T), mask.shape[::-1])
                edge_verts = np.array([v1, v2])
                cells = np.intersect1d(obj.V_df.at[v1, 'ncells'], obj.V_df.at[v2, 'ncells'])
                edge_rows.append({'pixels': pix, 'verts': edge_verts, 'cells': cells})

        if edge_rows:
            E_df = pd.DataFrame(edge_rows)
        else:
            E_df = pd.DataFrame(columns=['pixels', 'verts', 'cells'])

        # Edit V_df and C_df with edge information
        for v in range(0, len(obj.V_df)):
            for nv in obj.V_df.at[v, 'nverts']:
                edge_1 = np.argwhere((np.vstack(E_df['verts'])[:,0] == v)*(np.vstack(E_df['verts'])[:,1] == nv))
                edge_2 = np.argwhere((np.vstack(E_df['verts'])[:,1] == v)*(np.vstack(E_df['verts'])[:,0] == nv))

                if edge_1.size > 0:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], edge_1.ravel()[0])
                elif edge_2.size > 0:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], edge_2.ravel()[0])
                elif (v not in obj.C_df.at[0, 'nverts']) and (nv not in obj.C_df.at[0, 'nverts']):
                    # Create new edge
                    line = draw.line(obj.V_df.at[v, 'coords'][1], obj.V_df.at[v, 'coords'][0], obj.V_df.at[nv, 'coords'][1], obj.V_df.at[nv, 'coords'][0])
                    pix = np.ravel_multi_index(np.flip(line,axis=0), mask.shape[::-1])
                    edge_verts = np.array([v, nv])
                    cells = np.intersect1d(obj.V_df.at[v, 'ncells'], obj.V_df.at[nv, 'ncells'])
                    new_edge = pd.DataFrame({'pixels':[pix],'verts':[edge_verts],'cells':[cells]})
                    E_df = pd.concat([E_df, new_edge], ignore_index=True)
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], len(E_df) - 1)
                else:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], np.array([-1]))

        for c in range(1, len(obj.C_df)):
            c_verts = obj.C_df.at[c, 'nverts']

            if len(c_verts) > 1:
                c_coords = np.vstack(obj.V_df.loc[c_verts, 'coords'].to_list())
                c_coords = c_coords - np.mean(c_coords, axis=0)

                # Sort vertices in clockwise direction
                theta = np.mod(np.arctan2(c_coords[:,1], c_coords[:,0]), 2*np.pi)
                c_verts = c_verts[np.argsort(theta)]
                c_verts = np.append(c_verts, c_verts[0])


                if c not in obj.C_df.at[0, 'ncells']:
                    for v in range(0, len(c_verts)-1):
                        if (c_verts[v+1] in obj.V_df.at[c_verts[v], 'nverts']):
                            obj.C_df.at[c, 'edges'] = np.append(obj.C_df.at[c, 'edges'], np.intersect1d(obj.V_df.at[c_verts[v], 'edges'], obj.V_df.at[c_verts[v+1], 'edges']))
                        else:
                            obj.C_df.at[c, 'edges'] = np.append(obj.C_df.at[c, 'edges'], -1)

        return E_df

    def endpoints(self, image):
        # Define endpoint as pixel with only 1 4-connected neighbor
        # This requires the skeletonized image to be 4-connnected
        image = image.astype(int)
        k = np.array([[0,1,0],[1,0,1],[0,1,0]])
        neighborhood_count = ndi.convolve(image,k, mode='constant', cval=1)
        neighborhood_count[~image.astype(bool)] = 0
        return neighborhood_count == 1

    def segment_image(self, diameter=None, channels=[0,0], use_model='default'):
        """
        :param diameter: estimated diameter (px) for cells in image. If not specified, this will be estimated from the image
        :param channels: channels containing membrane and nuclear staining of image. 0 - Grayscale, 1 - R, 2 - G, 3 - B
        :param use_model: which Cellpose neural network to use. 'default' - Cyto2, 'custom' - custom trained model.
        :return: segmented image
        """
        from cellpose import models, utils, plot
        image = self.images.copy()
        image = image.astype(float)

        # Assuming membrane staining instead of cytoplasm, invert image before segmenting with Cellpose
        def normalise_image(image):
            image_norm = image.copy()
            ub = np.percentile(image, 99)
            lb = np.percentile(image, 1)
            image_norm[image_norm>ub] = ub
            image_norm[image_norm<lb] = lb
            image_norm = np.divide(image_norm-lb, ub - lb)
            return image_norm

        if channels != [0,0]:
            image[:,:,channels[0]-1] = 1-normalise_image(image[:,:,channels[0]-1])
            image[:,:,channels[1]-1] = normalise_image(image[:,:,channels[1]-1])
        else:
            image = normalise_image(image)

        if use_model == 'default':
            model = models.Cellpose(model_type='cyto2')
        elif use_model == 'custom':
            import pathlib
            src_path = str(pathlib.Path(__file__).parent.resolve())
            modeldir = f'{src_path}/cellpose_models/cellpose_residual_on_style_on_concatenation_off_train_folder_2022_03_24_00_26_36.748195'
            model = models.Cellpose(model_dir=modeldir, net_avg=False)
        else:
            return "Invalid use_model option. Available models are 'default', 'custom'."

        masks, flows, styles, diams = model.eval(image, diameter=diameter, channels=channels, progress=True)
        segmented_image = masks

        return segmented_image

    def polygon_perimeter(self):
        """

        Identify vertices and calculate polygon perimeter for each cell

        :return:
        """

        branchpoints = self.find_branch_points(self.masks==0)
        labels = np.unique(self.masks)
        labels = labels[labels!=0]
        res = pd.DataFrame(np.zeros([len(labels),1]), index=labels, columns=['polygon_perimeter'])

        # Build cell→branchpoints map in one pass over branchpoint coordinates,
        # replacing N separate binary_dilation calls (one per cell) with a single loop.
        bp_coords = np.argwhere(branchpoints)  # shape (M, 2): row, col
        cell_to_bp = {label: [] for label in labels}
        H, W = self.masks.shape
        for r, c in bp_coords:
            r0, r1 = max(0, r - 1), min(H, r + 2)
            c0, c1 = max(0, c - 1), min(W, c + 2)
            adj = np.unique(self.masks[r0:r1, c0:c1])
            for lbl in adj[adj != 0]:
                cell_to_bp[lbl].append((r, c))

        for label in labels:
            vertices = np.array(cell_to_bp[label])
            if vertices.shape[0] < 2:
                res.at[label, 'polygon_perimeter'] = 0
                continue
            # calculate polygon perimeter
            v_norm = vertices - np.mean(vertices, axis=0)
            theta = np.mod(np.arctan2(v_norm[:,1], v_norm[:,0]), 2*np.pi)
            vertices = vertices[np.argsort(theta),:]
            # Vectorized perimeter: roll vertices by 1 and compute all segment lengths at once
            v_rolled = np.roll(vertices, -1, axis=0)
            perim = np.sum(np.linalg.norm(vertices - v_rolled, axis=1))
            res.at[label, 'polygon_perimeter'] = perim
        centroids = pd.DataFrame(skimage.measure.regionprops_table(self.masks, properties=['label','centroid']))
        centroids.columns = ['label','centroid_y','centroid_x']
        centroids.index = centroids['label']
        res = pd.concat([res, centroids], axis=1)
        return res