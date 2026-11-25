'''
Fitting of the phase-distance distributions of the vortex matching simulations.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import f
from scipy.optimize import minimize

from histogram_plotting import HistogramPlot

#################################################
# HISTOGRAM FITTING #############################
#################################################

def wrap_phase_to_pi_pi(
    phases : np.ndarray,
):
    ''' Normalize the phases to the range [-pi, pi] '''
    return np.angle( np.exp(1j * phases) )

def wrap_phase_to_0_2pi(
    phases : np.ndarray,
):
    ''' Wrap phase to [0, 2*pi] '''
    phi_n             = wrap_phase_to_pi_pi(phases)
    phi_n[phi_n < 0] += 2 * np.pi
    return phi_n

def wrap_phase_to_05_05(
    phases : np.ndarray,
):
    ''' Wrap phase to [-0.5, 0.5] '''
    phases = wrap_phase_to_pi_pi(phases)
    return phases / (2 * np.pi)

def wrap_phase_to_00_10(
    phases : np.ndarray,
):
    ''' Wrap phase to [0, 1] '''
    phases = wrap_phase_to_0_2pi(phases)
    return phases / (2 * np.pi)

def _linear_phase(
    x_vals : np.ndarray,
    *params
):
    """ Linear function for phase fitting. """
    slope, intercept = params
    phase = slope * x_vals + intercept
    return phase

def _circular_errors(
    params   : tuple[float],
    x_samples: np.ndarray,
    p_samples: np.ndarray,
):
    ''' Compute circular errors for phase fitting. '''
    phi_fit = _linear_phase(x_samples, *params)
    errors  = wrap_phase_to_pi_pi(p_samples - phi_fit)
    return errors

def _circular_loss(
    params   : tuple[float],
    x_samples: np.ndarray,
    p_samples: np.ndarray,
):
    ''' Circular loss function for phase fitting. '''
    errors = _circular_errors(params, x_samples, p_samples)
    return np.sum(errors**2)

def _circular_r2(
    params   : tuple[float],
    x_samples: np.ndarray,
    p_samples: np.ndarray,
):
    ''' Circular R2 metric for phase fitting. '''

    # Predicted phases
    p_fit = _linear_phase(x_samples, *params)

    # Circular variance = 1 - R
    v_data = circular_variance(p_samples)
    v_res  = circular_variance(p_samples - p_fit)

    # Avoid division by zero
    v_data = max(v_data, np.finfo(float).eps)

    # Circular R2 (explained variance)
    r2 = 1 - v_res / v_data
    r2 = max(0.0, r2)

    return r2

def circular_mean(
    phases: np.ndarray
) -> float:
    """Compute the circular mean of an array of angles in radians."""

    # Compute average cosine and sine components
    phases = np.asarray(phases)
    c_mean = np.mean( np.cos(phases) )
    s_mean = np.mean( np.sin(phases) )

    # Compute angle of the mean vector
    return np.arctan2(s_mean, c_mean)

def circular_variance(
    phases : np.ndarray,
) -> float:
    ''' Compute the circular variance of an array of angles in radians. '''

    # Compute average cosine and sine components
    phases = np.asarray(phases)
    c_mean = np.mean( np.cos(phases) )
    s_mean = np.mean( np.sin(phases) )

    # Compute the resultant vector length (random walk length)
    rl = np.sqrt( c_mean**2 + s_mean**2 )

    # Compute circular variance
    var = 1 - rl

    return var

def _get_sample_values_from_histogram(
    x_vals   : np.ndarray,
    phi_vals : np.ndarray,
    histogram: np.ndarray,
    count_max: int = 1000,
):
    '''   Extract x and phi values from histogram. '''

    # Get the meshgrid for x and phi values
    X, PHI = np.meshgrid(x_vals, phi_vals, indexing='ij')
    counts = np.array(histogram, dtype=float)

    assert X.shape == PHI.shape == counts.shape, \
        "X, PHI, and counts must have the same shape"

    # Rescale counts to a maximum of count_max
    max_count = np.max(counts)
    if max_count > count_max:
        scale = count_max / max_count
        counts *= scale

    counts = np.round(counts).astype(int)

    # Flatten
    x   = X.ravel()
    phi = PHI.ravel()
    w   = counts.ravel()

    # Create lists to hold all x and phi values
    x_samples   = []
    phi_samples = []
    for xval, phi_val, count in zip(x, phi, w):
        if count <= 0:
            continue
        x_samples.extend(     [xval] * int(count))
        phi_samples.extend([phi_val] * int(count))

    # Convert to numpy arrays
    x_samples   = np.array(x_samples)
    phi_samples = np.array(phi_samples)

    return x_samples, phi_samples

def _print_fit_results(
    fit_results: dict
):
    """ Pretty print the results dictionary in a readable format. """


    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    print("Fit results:")
    print(json.dumps(fit_results, indent=2, default=convert))
    return

def _plot_fit_results(
    x_samples  : np.ndarray,
    fit_results: dict,
):
    ''' Plot the fitting performance of the linear phase fit. '''

    # Downsample the x_samples
    x_min     = np.min(x_samples)
    x_max     = np.max(x_samples)
    x_range   = x_max - x_min
    x_min     = x_min - 0.1 * x_range
    x_max     = x_max + 0.1 * x_range
    x_samples = np.linspace(x_min, x_max, 100)

    # Get the fit values
    fit_params = fit_results['params']
    phi_fit    = _linear_phase(x_samples, *fit_params)

    # Plot the data
    for n_cycle in [-2, -1, 0, 1, 2]:
        plt.plot(
            x_samples,
            phi_fit + n_cycle * 2 * np.pi,
            linestyle = '--',
            color     = 'k'
        )

    return

def _compute_fit_performance(
    params_opt : np.ndarray,
    x_samples  : np.ndarray,
    phi_samples: np.ndarray,
    phi_fit    : np.ndarray,
):
    ''' Evaluate the fit results. '''

    # Fit errors
    phase_errors = wrap_phase_to_pi_pi(phi_samples - phi_fit)
    ss_res       = np.sum( phase_errors**2 )

    # Phi covariance
    phi_mean = circular_mean(phi_samples)
    phi_diff = wrap_phase_to_pi_pi(phi_samples - phi_mean)
    ss_tot   = np.sum( phi_diff**2 )

    # Avoid division by zero
    ss_tot = max(ss_tot, np.finfo(float).eps)

    # Circular R2
    r2 = 1 - ss_res / ss_tot
    r2 = np.max( (0.0, r2) )

    # Circular LOSS
    circ_loss = _circular_loss(params_opt, x_samples, phi_samples)

    # Results
    fit_results = {
        'params' : params_opt,
        'loss'   : circ_loss,
        'r2'     : r2,
    }

    return fit_results

def _compute_line_fitting_samples(
    x_samples  : np.ndarray,
    phi_samples: np.ndarray,
    pos_slope  : bool = True,
    verbose    : bool = False,
    plot       : bool = False,
):
    ''' Compute line fitting values for histogram analysis. '''

    # Initial guess and bounds
    x_range   = np.max(x_samples) - np.min(x_samples)
    phi_range = np.max(phi_samples) - np.min(phi_samples)

    if not x_range or not phi_range:
        fit_results = {
            'params': [0.0, 0.0],
            'loss'  : np.inf,
            'r2'    : 0.0,
        }
        if verbose:
            print("\nWARNING: ZERO VARIANCE IN DATA. FIT NOT PERFORMED.")
            _print_fit_results(fit_results)
        return fit_results

    slope_max = 2 * phi_range / x_range
    slope_min = 0 if pos_slope else -slope_max

    slope_bounds = ( slope_min, slope_max )
    inter_bounds = ( -np.pi, np.pi )

    # Grid search
    n_search       = 20
    slope_grid     = np.linspace(slope_min, slope_max, n_search)
    intercept_grid = np.linspace(-np.pi, np.pi, n_search)

    best_loss   = np.inf
    best_r2     = 0.0
    best_params = None

    for slope in slope_grid:
        for intercept in intercept_grid:
            loss = _circular_loss([slope, intercept], x_samples, phi_samples)
            r2   = _circular_r2([slope, intercept], x_samples, phi_samples)
            if loss < best_loss:
                best_loss   = loss
                best_r2     = r2
                best_params = [slope, intercept]

    # Refine with optimizer
    result = minimize(
        _circular_loss,
        x0     = best_params,
        args   = (x_samples, phi_samples),
        bounds = [slope_bounds, inter_bounds],
    )

    if not result.success:
        fit_results = {
            'params': best_params,
            'loss'  : best_loss,
            'r2'    : best_r2,
        }
        if verbose:
            print("WARNING: FIT DID NOT CONVERGE.")
            _print_fit_results(fit_results)
        return fit_results

    # Evaluate the fit
    params_opt = result.x
    phi_fit    = _linear_phase(x_samples, *params_opt)
    phi_fit    = wrap_phase_to_pi_pi(phi_fit)

    # Results
    fit_results = _compute_fit_performance(
        params_opt  = params_opt,
        x_samples   = x_samples,
        phi_samples = phi_samples,
        phi_fit     = phi_fit
    )

    # Print
    if verbose:
        _print_fit_results(fit_results)

    if plot:
        _plot_fit_results(
            x_samples   = x_samples,
            fit_results = fit_results
        )

    return fit_results

def compute_line_fitting_histogram(
    x_vals   : np.ndarray,
    phi_vals : np.ndarray,
    histogram: np.ndarray,
    pos_slope: bool = True,
    plot     : bool = False,
    verbose  : bool = False,
):
    ''' Fit a linear relationship to the data.'''

    # Get sample values from histogram
    x_samples, phi_samples = _get_sample_values_from_histogram(
        x_vals    = x_vals,
        phi_vals  = phi_vals,
        histogram = histogram,
        count_max = 1000,
    )

    # Compute line fitting
    fit_results = _compute_line_fitting_samples(
        x_samples   = x_samples,
        phi_samples = phi_samples,
        pos_slope   = pos_slope,
        verbose     = verbose,
        plot        = plot,
    )

    return fit_results

#################################################
# LINEAR-CIRCULAR CORRELATION (ALL DATA) ########
#################################################

def _get_flattened_values(
    arr1_vals: np.ndarray,
    arr2_vals: np.ndarray,
    counts   : np.ndarray,
):
    ''' Get flattened arrays and weights from histogram. '''

    # Flatten the data into vectors
    ARR1, ARR2 = np.meshgrid(arr1_vals, arr2_vals, indexing='ij')

    assert ARR1.shape == ARR2.shape == counts.shape, \
        "ARR1, ARR2, and counts must have the same shape"

    values1 = ARR1.ravel()
    values2 = ARR2.ravel()
    weights = counts.ravel()

    return values1, values2, weights

def _get_weighted_mean(
    values : np.ndarray,
    weights: np.ndarray,
    sum_off: int = 0,
):
    ''' Compute the weighted mean of an array. '''
    wsum       = np.sum(weights) + sum_off
    mean_value = np.sum(weights * values) / wsum

    return mean_value

def _get_weighted_cov(
    values1: np.ndarray,
    values2: np.ndarray,
    weights: np.ndarray,
):
    ''' Compute the weighted covariance between two arrays. '''

    # Compute weighted means
    arr1_mean = _get_weighted_mean(values1, weights)
    arr2_mean = _get_weighted_mean(values2, weights)

    # Differences from the mean
    diff_1 = values1 - arr1_mean
    diff_2 = values2 - arr2_mean

    # Compute weighted covariance
    # NOTE: (sum-1) for unbiased estimate
    cov_12 = _get_weighted_mean(
        values  = diff_1 * diff_2,
        weights = weights,
        sum_off = -1
    )

    return cov_12

def _get_weighted_pearson_r(
    values1: np.ndarray,
    values2: np.ndarray,
    weights: np.ndarray,
):
    ''' Compute the covariance between two arrays. '''

    # NOTE: MATLAB CODE
    # function r = mycorr(a,b)
    #     a = a(:) - mean(a(:));
    #     b = b(:) - mean(b(:));
    #     denom = sqrt(sum(a.^2) * sum(b.^2));
    #     if denom == 0, r = 0; else, r = sum(a.*b) / denom; end
    # end

    # Compute weighted covariance
    cov_11 = _get_weighted_cov(values1, values1, weights)
    cov_22 = _get_weighted_cov(values2, values2, weights)
    cov_12 = _get_weighted_cov(values1, values2, weights)

    if not cov_11 or not cov_22:
        return 0.0

    return cov_12 / np.sqrt(cov_11 * cov_22)

def compute_linear_circular_correlation_from_distribution(
    x_vals     : np.ndarray,
    phi_vals   : np.ndarray,
    histogram  : np.ndarray,
    verbose    : bool = True,
):
    """
    Compute the linear-circular correlation coefficient
    between a linear variable x and a circular variable phi
    using a joint histogram of their occurrences.

    From Mardia (1979) and Johnson and Wehrly (1977).

    RHO ranges between zero and one (greater values = stronger correlation).
    RHO is invariant under a change of scale and origin of X
    RHO is invariant under a change of zero or sense of direction for Θ.
    """

    # NOTE: MATLAB CODE
    # function [rho, pval] = circLinCorr(x, theta)
    #     x = x(:); theta = theta(:);
    #     n = length(x);
    #     x = (x - mean(x)) / std(x);
    #     sx = sin(theta); cx = cos(theta);


    #     T = n * rho^2;
    #     pval = exp(-T/2); % chi2 with 2 df
    # end

    arr_x, arr_p, weights = _get_flattened_values(
        arr1_vals = x_vals,
        arr2_vals = phi_vals,
        counts    = histogram,
    )

    # Map to unit circle
    arr_c = np.cos(arr_p)
    arr_s = np.sin(arr_p)

    # Get pearson correlation coefficients
    rxs = _get_weighted_pearson_r(arr_x, arr_s, weights)
    rxc = _get_weighted_pearson_r(arr_x, arr_c, weights)
    rcs = _get_weighted_pearson_r(arr_c, arr_s, weights)

    # Compute linear-circular correlation
    num = rxs**2 + rxc**2 - 2 * rxs * rxc * rcs
    den = 1 - rcs**2

    # Avoid division by zero
    den = max( np.finfo(float).eps, den )

    # Correlation coefficient
    rho = np.sqrt( num / den )

    # NOTE:
    # If X and Θ are indepenent and X is normally distributed
    # Under Null hypothesis of rho = 0
    # (n−3) * Rho**2 / (1 − Rho**2) ∼ F(2,n−3)

    # Compute the F statistic
    n     = np.sum(histogram)
    f_val = (n - 3) * rho**2 / (1 - rho**2)

    # Compute p-value from the tail (1 - CDF)
    pval = 1 - f.cdf(f_val, dfn=2, dfd=n - 3)

    # Print results
    if verbose:
        fit_results = {
            "rho" : f"{rho:.5e}",
            "pval": f"{pval:.5e}",
        }
        print("Linear-Circular Correlation from Distribution:")
        print(json.dumps(fit_results, indent=2))

    return rho, pval

#################################################
# LINEAR-CIRCULAR CORRELATION (MEAN ANGLE) ######
#################################################

def _get_mean(
    values : np.ndarray,
    sum_off: int = 0,
):
    ''' Compute the mean of an array. '''
    num = np.sum(values)
    den = len(values) + sum_off
    den = max(den, np.finfo(float).eps)
    return num / den

def _get_cov(
    values1: np.ndarray,
    values2: np.ndarray,
):
    ''' Compute the covariance between two arrays. '''

    # Compute means
    arr1_mean = _get_mean(values1)
    arr2_mean = _get_mean(values2)

    # Differences from the mean
    diff_1 = values1 - arr1_mean
    diff_2 = values2 - arr2_mean

    # Compute covariance
    # NOTE: (sum-1) for unbiased estimate
    cov_12 = _get_mean(
        values  = diff_1 * diff_2,
        sum_off = -1
    )

    return cov_12

def _get_z_val(
    values : np.ndarray,
):
    ''' Compute the z value of an array. '''

    v_mean = _get_mean(values)
    v_std  = np.sqrt( _get_cov(values, values) )

    # Avoid division by zero
    if v_std == 0:
        return np.zeros_like(values)

    # Compute z value
    z_val  = (values - v_mean) / v_std

    return z_val

def _get_pearson_r(
    values1: np.ndarray,
    values2: np.ndarray,
):
    ''' Compute the covariance between two arrays. '''

    # NOTE: MATLAB CODE
    # function r = mycorr(a,b)
    #     a = a(:) - mean(a(:));
    #     b = b(:) - mean(b(:));
    #     denom = sqrt(sum(a.^2) * sum(b.^2));
    #     if denom == 0, r = 0; else, r = sum(a.*b) / denom; end
    # end

    # Compute covariance
    cov_11 = _get_cov(values1, values1)
    cov_22 = _get_cov(values2, values2)
    cov_12 = _get_cov(values1, values2)

    if not cov_11 or not cov_22:
        return 0.0

    return cov_12 / np.sqrt(cov_11 * cov_22)

def _get_weighted_circular_mean(
    phases : np.ndarray,
    weights: np.ndarray,
    sum_off: int = 0,
):
    ''' Compute the weighted mean of an array. '''
    wsum = np.sum(weights) + sum_off

    # Convert to unit circle representation
    c_vals = np.cos(phases)
    s_vals = np.sin(phases)

    # Weighted average components
    mean_c = np.sum(weights * c_vals) / wsum
    mean_s = np.sum(weights * s_vals) / wsum

    # Compute angle of the mean vector
    mean_value = np.arctan2(mean_s, mean_c)

    return mean_value

def _get_mean_histogram_angles(
    x_vals   : np.ndarray,
    phi_vals : np.ndarray,
    histogram: np.ndarray,
):
    ''' Bin the x values and compute circular means of theta in each bin. '''

    # NOTE: MATLAB CODE
    # function [binCenters, meanAngles, binIdxCells] = circBinAvgDetailed(x, theta, nbins)
    #     edges = linspace(min(x), max(x), nbins+1);
    #     binCenters = (edges(1:end-1) + edges(2:end)) / 2;
    #     meanAngles = nan(1, nbins);
    #     binIdxCells = cell(1, nbins);
    #     for k = 1:nbins
    #         idx = x >= edges(k) & x < edges(k+1);
    #         binIdxCells{k} = find(idx);
    #         if any(idx)
    #             meanAngles(k) = circ_mean(theta(idx));
    #         end
    #     end
    # end

    mean_angles = np.array(
        [
            _get_weighted_circular_mean(
                phases  = phi_vals,
                weights = histogram[x_ind, :],
            )
            for x_ind, _x_val in enumerate(x_vals)
        ]
    )
    return mean_angles

def compute_linear_circular_correlation_from_mean(
    x_vals     : np.ndarray,
    phi_vals   : np.ndarray,
    histogram  : np.ndarray,
    verbose    : bool = True,
):
    """
    Compute the linear-circular correlation coefficient
    between a linear variable x and a circular variable phi
    using a joint histogram of their occurrences.
    NOTE: This function uses the mean angles in each x bin.

    From Mardia (1979) and Johnson and Wehrly (1977).

    RHO ranges between zero and one (greater values = stronger correlation).
    RHO is invariant under a change of scale and origin of X
    RHO is invariant under a change of zero or sense of direction for Θ.
    """

    # NOTE: MATLAB CODE
    # function [rho, pval] = circLinCorr(x, theta)
    #     x = x(:); theta = theta(:);
    #     n = length(x);
    #     x = (x - mean(x)) / std(x);
    #     sx = sin(theta); cx = cos(theta);


    #     T = n * rho^2;
    #     pval = exp(-T/2); % chi2 with 2 df
    # end

    # Get mean angles from histogram
    arr_x = _get_z_val(x_vals)
    arr_p = _get_mean_histogram_angles(
        x_vals    = x_vals,
        phi_vals  = phi_vals,
        histogram = histogram,
    )

    arr_c = np.cos(arr_p)
    arr_s = np.sin(arr_p)

    # Get pearson correlation coefficients
    rxs = _get_pearson_r(arr_x, arr_s)
    rxc = _get_pearson_r(arr_x, arr_c)
    rcs = _get_pearson_r(arr_c, arr_s)

    # Compute linear-circular correlation
    num = rxs**2 + rxc**2 - 2 * rxs * rxc * rcs
    den = 1 - rcs**2

    # Avoid division by zero
    den = max( np.finfo(float).eps, den )

    # Correlation coefficient
    rho = np.sqrt( num / den )

    # NOTE:
    # If X and Θ are indepenent and X is normally distributed
    # Under Null hypothesis of rho = 0
    # (n−3) * Rho**2 / (1 − Rho**2) ∼ F(2,n−3)

    # Compute the F statistic
    n     = len(x_vals)
    f_val = (n - 3) * rho**2 / (1 - rho**2)

    # Compute p-value from the tail (1 - CDF)
    pval = 1 - f.cdf(f_val, dfn=2, dfd=n - 3)

    # Print results
    if verbose:
        fit_results = {
            "rho" : f"{rho:.5e}",
            "pval": f"{pval:.5e}",
        }
        print("Linear-Circular Correlation from Mean:")
        print(json.dumps(fit_results, indent=2))

    return rho, pval

#################################################
# HISTOGRAM PLOTTING ############################
#################################################

class HistogramPlotFitting(HistogramPlot):

    def __init__(
        self,
        histogram: np.ndarray,
        x_vals   : np.ndarray,
        phi_vals : np.ndarray,
        verbose  : bool = True,
    ):
        ''' Initialize the histogram plot with fitting capabilities. '''
        HistogramPlot.__init__(
            self,
            histogram = histogram,
            x_vals    = x_vals,
            phi_vals  = phi_vals,
        )

        # Compute fitting results
        self.fit_results  = self.compute_fitting_line(
            plot    = False,
            verbose = verbose,
        )
        self.corr_results = self.compute_linear_circular_correlation(
            verbose = verbose,
        )
        return

    ##################
    # FITTING ########
    ##################

    def compute_fitting_line(
        self,
        plot   : bool = True,
        verbose: bool = True,
    ):
        ''' Compute the fitting line on the histogram data. '''

        self.fit_results = compute_line_fitting_histogram(
            x_vals    = self.x_vals,
            phi_vals  = self.phi_vals,
            histogram = self.histogram,
            pos_slope = True,
            plot      = plot,
            verbose   = verbose,
        )

        return self.fit_results

    def compute_linear_circular_correlation(
        self,
        verbose: bool = True,
    ):
        ''' Compute the linear-circular correlation on the histogram data. '''

        rho_m, pval_m = compute_linear_circular_correlation_from_mean(
            x_vals      = self.x_vals,
            phi_vals    = self.phi_vals,
            histogram   = self.histogram,
            verbose     = verbose,
        )

        rho_d, pval_d = compute_linear_circular_correlation_from_distribution(
            x_vals      = self.x_vals,
            phi_vals    = self.phi_vals,
            histogram   = self.histogram,
            verbose     = verbose,
        )
        self.corr_results = {
            "rho_mean" : rho_m, "pval_mean": pval_m,
            "rho_dist" : rho_d, "pval_dist": pval_d
        }

        return self.corr_results

    ##################
    # PLOTS ##########
    ##################

    def _plot_fit_line(self):
        ''' Plot the fitting line on the current plot. '''
        _plot_fit_results(
            x_samples   = self.x_edges,
            fit_results = self.fit_results
        )
        return

    def _plot_fit_line_polar(self):
        ''' Plot the fitting line on the current polar plot. '''
        a, b = self.fit_results["params"]

        R_line  = np.linspace(0.0, self.x_lims[1], 400)
        Phi_fit = wrap_phase_to_pi_pi(a * R_line + b)

        ax = plt.gca()
        ax.plot(
            R_line * np.cos(Phi_fit),
            R_line * np.sin(Phi_fit),
            linewidth = 2.0,
            linestyle = "--",
            color     = "black",
            label     = r"fit: $z(R)=R\,e^{i(aR+b)}$"
        )
        ax.legend(frameon=False)

        return

    def plot_histogram(
        self,
        max_density = None,
    ):
        """Plot the current histogram as a heatmap."""
        fig, ax = super().plot_histogram(max_density)

        # Overlay fitted curve
        _plot_fit_results(
            x_samples   = self.x_edges,
            fit_results = self.fit_results
        )

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

        fig, ax = super().plot_histogram_polar(
            max_density = max_density,
            filter      = filter
        )

        # Overlay fitted curve Phi(R) = a*R + b
        self._plot_fit_line_polar()

        return fig, ax

##################################################
# FULL ANALYSIS ##################################
##################################################

def run_analysis(
    histogram : np.ndarray,
    x_vals    : np.ndarray,
    phi_vals  : np.ndarray,
    verbose   : bool = True,
):
    ''' Run the full  analysis. '''

    # Get histogram
    hist_plot = HistogramPlotFitting(
        histogram = histogram,
        x_vals    = x_vals,
        phi_vals  = phi_vals,
        verbose   = verbose,
    )

    # Plot the histogram
    hist_plot.plot_histogram(
        max_density = 0.20,
    )

    # Plot polar histogram
    hist_plot.plot_histogram_polar(
        max_density = 0.20,
    )

    plt.show()

    return
