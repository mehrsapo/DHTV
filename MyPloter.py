import numpy as np
import torch

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_points(ax, points=None, values=None, simplices=None,
                grid=False, axis='on', alpha=1, color='black', pane_color=(1.0, 1.0, 1.0, 1.0),
                title=None, xlabel='', ylabel='', zlabel='',
                point_names=None, print_point_names=True, s=1):
    """
    This functions plot points (2D or 3D (with values)) w/w.o. delaunay simplices
    """

    if values is not None:
        if simplices is None:
            ax.scatter(points[:, 0], points[:, 1], values, color=color, alpha=alpha, s=s)
            ax.w_xaxis.set_pane_color(pane_color)
            ax.w_yaxis.set_pane_color(pane_color)
            ax.w_zaxis.set_pane_color(pane_color)

        ax.set_zlabel(zlabel)
        if point_names is not None:
            if print_point_names:
                for i, name in enumerate(point_names):
                    ax.text(points[i, 0] + .005, points[i, 1] + .005, values[i] + .005, name, weight="bold", fontsize=11)

    if values is None:
        if simplices is None:
            ax.scatter(points[:, 0], points[:, 1], color=color, alpha=alpha)

        else:
            ax.triplot(points[:, 0], points[:, 1], simplices)
            ax.scatter(points[:, 0], points[:, 1], color=color)

        ax.set_aspect('equal')

        if point_names is not None:
            if print_point_names:
                for i, name in enumerate(point_names):
                    ax.text(points[i, 0] + .005, points[i, 1] + 0.005, name, weight="bold", fontsize=11)


    ax.grid(grid)
    ax.axis(axis)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def color_simplices(ax, selected, simplices_points, points, point_names=None,
                    selected_fcolors=None, print_simplex_name=True, title=''):
    """
    This function fills selected simplices with chosen colors (if not, random)
    """
    simplices_centers = np.sum(points[simplices_points], axis=1) / 3
    selected_centers = simplices_centers[selected]
    selected_simplices = simplices_points[selected]
    if selected_fcolors is None:
        selected_fcolors = np.random.permutation(len(selected))
    else:
        selected_fcolors = np.array(selected_fcolors)
    # create a colormap with a single color
    cmap = "Pastel1"
    ax.tripcolor(points[:, 0], points[:, 1], selected_simplices, facecolors=selected_fcolors, cmap=cmap)
    simplices_names = ['$P_{' + str(i) + '}$' for i in selected]
    if print_simplex_name:
        for i, name in enumerate(simplices_names):
            ax.text(selected_centers[i, 0], selected_centers[i, 1], name, fontsize=12)

    ax = plot_points(ax, points=points, title=title, simplices=simplices_points,
                     xlabel='$x^{(1)}$', ylabel='$x^{(2)}$', point_names=point_names)

    return ax

def simplex_neighbor(tri, i):
    """
    This function gives indices of (valid, non -1) neighbors of simplex P_i
    """
    neigh = tri.neighbors[i]
    neigh = list(neigh[neigh>=0])
    return neigh

def fill_selected_neighbors(ax, tri, i, points, title=''):
    """
    This function fills simplex P_i with a color and its neighbors with a different color :D
    """
    assert i > -1
    selected = [i]
    neigh = simplex_neighbor(tri, i)
    selected_all = selected + neigh
    color_simplices(ax, selected_all, tri.simplices, points, selected_fcolors=[0] + [1] * len(neigh), title=title)
    return ax


def plot_3d_cpwl(ax, tri, values, point_names=None, color='black', face_color='black'):

    for s in tri.simplices:
        coords = tri.grid_points[s]
        verts = np.concatenate((coords, values[s][:, np.newaxis]), axis=1)
        vert_tuple = list(map(tuple, verts))
        fc = get_normal_facecolor(tri.lat_coeffs)
        srf = Poly3DCollection(vert_tuple, alpha=0.5, facecolor=face_color, edgecolors='b', linewidths=0.5)

        ax.add_collection3d(srf)


    ax = plot_points(ax, points=tri.grid_points, values=values, title='3D view, points and values', xlabel='$x^{(1)}$',
        ylabel='$x^{(2)}$', zlabel='$y$', point_names=point_names, s=5, color=color)

    return ax


import plotly.graph_objects as go
import copy


def map_val2color2d(val, vmin, vmax):
    """ Map the normalized 2D value to a corresponding R, B channel.
    val, vmin, vmax: 2d array
    """
    if vmin[0] > vmax[0] or vmin[1] > vmax[1]:
        raise ValueError('incorrect relation between vmin and vmax')

    t = np.zeros(2)
    if not np.allclose(vmin[0], vmax[0]):
        t[0] = (val[0] - vmin[0]) / float((vmax[0] - vmin[0]))  # normalize val
    if not np.allclose(vmin[1], vmax[1]):
        t[1] = (val[1] - vmin[1]) / float((vmax[1] - vmin[1]))  # normalize val

    R, G = t[1], t[0]
    B = 0.4
    return 'rgb(' + '{:d}'.format(int(R * 255 + 0.5)) + ',' + '{:d}'.format(int(G * 255 + 0.5)) + \
           ',' + '{:d}'.format(int(B * 255 + 0.5))  + ')'
    #return (R, G, B, 0.5)



def map_array2color2d(array, min=None, max=None):
    """ """
    if min is None:
        min = array.amin(axis=0)
    if max is None:
        max = array.amax(axis=0)

    return np.array([map_val2color2d(val, min, max) for val in array])


def get_normal_facecolor(affine_coeff):
    """ Get facecolor of triangles according to
    their normals.
    """
    max = np.array([1.75, 1.75])
    facecolor = map_array2color2d(-affine_coeff, min=-max, max=max)

    return facecolor

def get_line(x, y, z):
    """ """
    line = go.Scatter3d(x=x, y=y, z=z,
                        marker=dict(size=1, symbol='circle',
                                    color="rgb(84,48,5)"),
                        line=dict(color="rgb(84,48,5)",
                                  width=8),
                        opacity=0.9)

    return line


def add_normals_plot(plot_data, tri_reg, normals_scale=0.02):

    """ Get normals and add normal 3Dscatter plot to plot_data.

    Args:
        scale: multiplicative factor of length of normal vectors
    """

    x_std_centers = torch.from_numpy(tri_reg.valid_simplices_centers)
    x_values= torch.from_numpy(tri_reg.evaluate(x_std_centers))
    affine_coeff = torch.from_numpy(tri_reg.lat_coeffs)

    for i in range(x_std_centers.size(0)):
        normal_x = torch.tensor([x_std_centers[i, 0], x_std_centers[i, 0] - affine_coeff[i, 0] * normals_scale])
        normal_y = torch.tensor([x_std_centers[i, 1], x_std_centers[i, 1] - affine_coeff[i, 1] * normals_scale])
        normal_z = torch.tensor([x_values[i], x_values[i].add(1. * normals_scale)])

        normal_x, normal_y, normal_z = normal_x.numpy(), normal_y.numpy(), normal_z.numpy()

        normal_i = get_line(x=normal_x, y=normal_y, z=normal_z)
        plot_data.append(normal_i)

    return plot_data


def plot_with_gradient_map(tri_reg, a=1.2, b=1.4, c=1.3, lmbda=None, save_plot=False, title='', up=False, selected=None):

    if selected is None:
        selected = list(range(tri_reg.n_simplices))

    simplices = tri_reg.valid_simplices
    x_std = tri_reg.grid_points
    z = tri_reg.grid_values
    i = simplices[selected, 0]
    j = simplices[selected, 1]
    k = simplices[selected, 2]
    opacity = 1
    fc = get_normal_facecolor(tri_reg.lat_coeffs[selected])

    data = [go.Mesh3d(x=x_std[:, 0], y=x_std[:, 1], z=z, i=i, j=j, k=k, facecolor=fc, opacity=opacity)]

    # data = add_normals_plot(data, tri_reg)
    fig = go.Figure(data=data)

    '''fig.update_traces(lighting=dict(ambient=0.1, diffuse=1, specular=0.1),
                      lightposition=dict(x=0, y=0, z=4),
                      selector=dict(type='mesh3d'))'''

    # view = '3D'  # default view

    # assert view in ['up', 'side', '3D', '3D_2']

    ax_dict = dict(linecolor='#000000', linewidth=4, showgrid=False,
                   showticklabels=False, gridcolor='#000000', gridwidth=0.3,
                   title=dict(font=dict(size=35)), showbackground=True)

    fig_dict = dict(
        scene_aspectmode='data',
        title= '$\lambda =' + str(lmbda) + ' - \mathrm{' +  title + '}$ ',
        scene=dict(
            xaxis=copy.deepcopy(ax_dict),
            yaxis=copy.deepcopy(ax_dict),
            zaxis=copy.deepcopy(ax_dict),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        font=dict(size=30),
    )

    '''if view == 'side':
        fig_dict['scene']['camera']['eye'] = dict(x=1.2, y=0.3, z=0.4)
    elif view == '3D':
        fig_dict['scene']['camera']['eye'] = dict(x=1.2, y=1.4, z=1.3)
    elif view == '3D_2':
        fig_dict['scene']['camera']['eye'] = dict(x=1.0, y=-2.0, z=0.3)
    elif view == 'up':
        fig_dict['scene']['zaxis']['visible'] = False
        fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=3)
        fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)'''


    fig_dict['scene']['camera']['eye'] = dict(x=a, y=b, z=c)
    if up:
        fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)

    fig.update_layout(**fig_dict)

    if save_plot:
        fig.write_image("face/correct_lmbda" + str(lmbda) + ".png")

    fig.show()

