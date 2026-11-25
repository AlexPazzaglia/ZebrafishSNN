'''
Plotting of the phase-distance distributions of the vortex matching simulations.
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.image import NonUniformImage

from scipy.ndimage import gaussian_filter

class HistogramPlot:

    def __init__(
        self,
        histogram : np.ndarray,
        x_vals    : np.ndarray,
        phi_vals  : np.ndarray,
    ):
        """ Histogram plot """

        if hasattr(self, 'initialized') and self.initialized:
            return

        self.histogram = histogram
        self.x_vals    = x_vals
        self.phi_vals  = phi_vals

        # X bins
        xv = self.x_vals
        if len(xv) >= 2:
            dx = ( xv[1] - xv[0] ) / 2
        else:
            dx = 0.05

        self.x_bins  = len(xv)
        self.x_step  = 2 * dx
        self.x_edges = np.concatenate( ( [xv[0] - dx], xv + dx ) )
        self.x_lims  = (self.x_edges[0], self.x_edges[-1])
        self.x_range = self.x_lims[1] - self.x_lims[0]

        # Phi bins
        pv = self.phi_vals
        if len(pv) >= 2:
            dp = ( pv[1] - pv[0] ) / 2
        else:
            dp = 0.05

        self.phi_bins  = len(pv)
        self.phi_step  = 2 * dp
        self.phi_edges = np.concatenate( ( [pv[0] - dp], pv + dp ) )
        self.phi_lims  = ( self.phi_edges[0], self.phi_edges[-1] )
        self.phi_range = self.phi_lims[1] - self.phi_lims[0]

        # Checks
        assert self.histogram.shape == (self.x_bins, self.phi_bins), \
            f'Histogram shape {self.histogram.shape} != {(self.x_bins, self.phi_bins)}'

        # Initialized
        self.initialized = True

        return

    ##################
    # UTILS ##########
    ##################

    def _get_normalized_histogram(
        self,
        histogram  : np.ndarray,
        max_density: float = None,
    ):
        """ Get the normalized histogram """

        # Normalize by number of iterations per distance
        np_per_dist = np.sum(histogram, axis=1)
        hist_norm   = histogram / np_per_dist[:, np.newaxis]

        # Default: Maximum
        if max_density is None:
            max_density = np.max(hist_norm)

        # Normalize by maximum density
        hist_norm = hist_norm / max_density

        return hist_norm

    def _get_filtered_histogram(
        self,
        histogram: np.ndarray,
        sigma    : float = 1.0,
    ):

        hist_max   = np.max(histogram)
        hist_min   = np.min(histogram)
        hist_range = hist_max - hist_min

        histogram = gaussian_filter(
            input = histogram,
            sigma = sigma,
        )

        new_max   = np.max(histogram)
        new_min   = np.min(histogram)
        new_range = new_max - new_min

        histogram = (
            hist_min +
            hist_range * (histogram - new_min) / new_range
        )
        return histogram

    def _decorate_histogram_plot(self):
        ''' Decorate the histogram plot '''

        # Add dashed lines at multiples of pi
        for i in range(-1, 6):
            plt.axhline(i * np.pi, color='white', linestyle='--', linewidth=0.8)

        # Put ylabels and ticks at multiples of pi
        plt.yticks(
            ticks  = np.arange(-np.pi, 5*np.pi+1, np.pi),
            labels = [
                r'$-\pi$', r'$0$', r'$\pi$',
                r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$',
            ]
        )

        # Increas thickness of ticks
        plt.tick_params(axis='both', which='major', width=1.5)

        # Label axes
        plt.xlabel("FB distance (BL)", fontsize=14)
        plt.ylabel("Phase difference $\\Delta \\Phi$", fontsize=14)
        plt.title("$\\Delta \\Phi = \\Phi_L - \\Phi_F$", fontsize=16)

        # Set limits
        plt.xlim(self.x_lims)
        plt.ylim(self.phi_lims)

        return

    def _plot_distance_circles(
        self,
        ax : plt.Axes,
    ):
        ''' Plot distance guide circles '''
        for r_vals in np.unique(self.x_vals):
            if r_vals <= 0:
                continue
            circ = Circle(
                xy        = (0.0, 0.0),
                radius    = r_vals,
                fill      = False,
                linewidth = 0.8,
                alpha     = 0.3
            )
            ax.add_patch(circ)
        return

    ##################
    # DATA HANDLING ##
    ##################

    def save_histogram_counts(
        self,
        save_folder   : str,
    ):
        ''' Save the collected data '''

        # Convert to DataFrame
        histogram_df = pd.DataFrame(
            data    = self.histogram.T,
            columns = self.x_vals,
            index   = self.phi_vals,
        )

        # Create folder if it does not exist
        os.makedirs(save_folder, exist_ok=True)

        # Save to csv
        print(f'Saving histogram data to {save_folder}/phase_matching_histogram.csv')
        histogram_df.to_csv(f'{save_folder}/phase_matching_histogram.csv')

        return

    ##################
    # PLOTS ##########
    ##################

    def plot_histogram(
        self,
        max_density: float = None,
    ):
        """Plot the current histogram as a heatmap."""

        # Normalize the histogram
        histogram = self._get_normalized_histogram(
            histogram   = self.histogram,
            max_density = max_density,
        )

        # Create figure
        figname = 'phase_matching_histogram'
        fig     = plt.figure(figname, figsize=(8,6))
        ax      = fig.add_subplot(111)

        # Plot the histogram
        mesh_x, mesh_Y = np.meshgrid(self.x_edges, self.phi_edges)
        pcm = ax.pcolormesh(
            mesh_x,
            mesh_Y,
            histogram.T,
            cmap    = 'hot',
            shading = 'auto',
            vmin    = 0.0,
            vmax    = 1.0,
        )
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Phase matching ratio", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Decorate the plot
        self._decorate_histogram_plot()

        return fig, ax

    def plot_histogram_non_uniform_image(
        self,
        max_density: float = None,
    ):
        ''' Plot the histogram using NonUniformImage '''

        # Normalize the histogram
        histogram = self._get_normalized_histogram(
            histogram   = self.histogram,
            max_density = max_density,
        )

        # Create figure
        figname = 'phase_matching_histogram_non_uniform_image'
        fig     = plt.figure(figname, figsize=(8,6))
        ax      = fig.add_subplot(111, xlim=self.x_lims, ylim=self.phi_lims)

        # Plot the histogram
        im = NonUniformImage(ax, interpolation='bilinear')
        im.set_data(self.x_vals, self.phi_vals, histogram.T)
        ax.add_image(im)

        # Decorate the plot
        self._decorate_histogram_plot()

        return fig, ax

    def plot_histogram_3d(
        self,
        max_density: float = None,
        filter      : bool = True,
    ):
        ''' Plot the histogram in 3D '''

        # Normalize the histogram
        histogram = self._get_normalized_histogram(
            histogram   = self.histogram,
            max_density = max_density,
        )

        # Filter the 3D histogram, preserving the range
        if filter:
            tag       = 'filtered'
            histogram = self._get_filtered_histogram(
                histogram = histogram,
                sigma     = 1.0,
            )
        else:
            tag = 'unfiltered'

        ## 3D figure
        figname = f'phase_matching_histogram_3d_{tag}'
        fig     = plt.figure(figname, figsize=(8,6))
        ax      = fig.add_subplot(111, projection='3d')

        # Plot the surface
        mesh_x, mesh_Y = np.meshgrid(self.x_vals, self.phi_vals)

        ax.plot_surface(
            mesh_x,
            mesh_Y,
            histogram.T,
            cmap='hot'
        )

        # Labels
        x0, x1 = self.x_lims
        p0, p1 = self.phi_lims
        xmax = x1 + 0.05 * self.x_range
        pmax = p1 + 0.05 * self.phi_range

        ax.set_xlabel("FB distance (BL)", fontsize=14)
        ax.set_xticks(np.arange(x0, xmax, step=1))
        ax.set_yticks(np.arange(p0, pmax, step=np.pi))
        ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'])
        ax.set_ylabel("Phase difference $\\Delta \\Phi$", fontsize=14)
        ax.set_zlabel("Phase matching ratio", fontsize=14)
        ax.set_title("$\\Delta \\Phi = \\Phi_L - \\Phi_F$", fontsize=16)
        ax.set_zlim(0, 1)

        return fig, ax

    def plot_histogram_polar(
        self,
        max_density: float = None,
        filter      : bool = True,
    ):
        """
        Plot a smoothed density of points in the complex plane by
        Gaussian-smoothing the (R, Phi) histogram and mapping it to (x,y).
        """

        # Normalize
        hist = self._get_normalized_histogram(
            histogram   = self.histogram,
            max_density = max_density,
        )

        # Filter
        if filter:
            tag  = 'filtered'
            hist = self._get_filtered_histogram(
                histogram = hist,
                sigma     = 1.0,
            )
        else:
            tag = 'unfiltered'

        # Build edge grids in cartesian coordinates
        R, P = np.meshgrid(
            self.x_edges,
            self.phi_edges,
            indexing="ij"
        )
        X = R * np.cos(P)
        Y = R * np.sin(P)

        # Create figure
        figname = f'phase_matching_histogram_polar_{tag}'
        fig     = plt.figure(figname, figsize=(8,6))
        ax      = fig.add_subplot(111)

        # Plot the density
        pcm = ax.pcolormesh(
            X, Y, hist,
            cmap    = 'hot',
            shading = "auto",
            vmin    = 0.0,
            vmax    = 1.0
        )
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Point density (smoothed)", fontsize=12)

        # Distance guide circles
        self._plot_distance_circles(ax=ax)

        # Decorate
        ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.axvline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\Re\{z\}$", fontsize=12)
        ax.set_ylabel(r"$\Im\{z\}$", fontsize=12)
        ax.set_title(r"Smoothed density in complex plane (from $(R,\Delta\Phi)$)", fontsize=14)

        # Limits padded by 5% of max R
        r_max = self.x_lims[1]
        pad = 0.05 * r_max
        ax.set_xlim(-r_max - pad, r_max + pad)
        ax.set_ylim(-r_max - pad, r_max + pad)

        plt.tight_layout()

        return fig, ax
