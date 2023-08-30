"""
These functions are used to plot reachable sets represented vertices

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause


"""

import sys
import numpy as np
import mpl_toolkits.mplot3d as a3
from scipy.spatial import ConvexHull
import itertools
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def plot_polytope2d(set_vs, ax, color='r',alpha=1.0, edgecolor='k', linewidth=1.0, zorder=1):
    """
    Function to plot 2-dimensional polytope

    Parameters:
        set_vs (np.ndarray): Vertices of the set
        ax (AxesSubplot): AxesSubplot
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges
        zorder (int): Plotting order

    """
    try: # 2 dimensional hull
        hull = ConvexHull(set_vs)
        fs, ps  = hull.equations, hull.points
        fps = (abs(np.dot(fs[:,0:2], ps.T) + fs[:,[2]])<=1e-6).astype(int)
        bools = np.sum(fps, axis=0)==2
        ps_new = ps[bools,:]
        fps, indx = np.unique(fps[:, bools], return_index=True, axis=1)
        ps_new = ps_new[indx, :]
        ps_adj = np.dot(fps.T, fps) # adjacency between points
        indx_adj = [np.nonzero(arr==1)[0] for arr in ps_adj]

        indx_ps_new = [0] # along the edge
        curr_point = 0
        for n in range(len(indx_adj)-1):
            next_idnx0, next_indx1 = indx_adj[curr_point][0], indx_adj[curr_point][1]
            if next_idnx0 not in indx_ps_new:
                indx_ps_new.append(next_idnx0)
                curr_point = next_idnx0
            elif next_indx1 not in indx_ps_new:
                indx_ps_new.append(next_indx1)
                curr_point = next_indx1
            else:
                sys.exit('Wrong points')

        ps_final = ps_new[indx_ps_new, :]
    except: # 0 or 1 dimensional hull
        ps_final = set_vs

    poly = Polygon(ps_final, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection(PatchCollection([poly], match_original=True,zorder=zorder))


def plot_polytope3d(set_vs, ax, color='r',alpha=1.0, edgecolor='k', linewidth=1.0):
    """
    Function to plot 3-dimensional polytope

    Parameters:
        set_vs (np.ndarray): Vertices of the set
        ax (AxesSubplot): AxesSubplot
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges

    """
    hull = ConvexHull(set_vs)
    faces = hull.simplices
    for s in faces:
        sq = [
            [set_vs[s[0], 0], set_vs[s[0], 1], set_vs[s[0], 2]],
            [set_vs[s[1], 0], set_vs[s[1], 1], set_vs[s[1], 2]],
            [set_vs[s[2], 0], set_vs[s[2], 1], set_vs[s[2], 2]]
        ]
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(color)
        f.set_edgecolor(edgecolor)
        f.set_alpha(alpha)
        f.set_linewidth(linewidth)
        ax.add_collection3d(f)


def plot_box3d(lbs, ubs, ax, color='r', alpha=1.0, edgecolor='k', linewidth=1.0):
    """
    Function to plot 3-dimensional box

    Parameters:
        lbs (list): Lower bounds of the box
        ubs (list): Upper bounds of the box
        ax (AxesSubplot): AxesSubplot
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges

    """

    V = []
    for i in range(len(lbs)):
        V.append([lbs[i], ubs[i]])

    vs = np.array(list(itertools.product(*V)))
    faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    for s in faces:
        sq = [
            [vs[s[0], 0], vs[s[0], 1], vs[s[0], 2]],
            [vs[s[1], 0], vs[s[1], 1], vs[s[1], 2]],
            [vs[s[2], 0], vs[s[2], 1], vs[s[2], 2]],
            [vs[s[3], 0], vs[s[3], 1], vs[s[3], 2]]
        ]
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(color)
        f.set_edgecolor(edgecolor)
        f.set_alpha(alpha)
        f.set_linewidth(linewidth)
        ax.add_collection3d(f)