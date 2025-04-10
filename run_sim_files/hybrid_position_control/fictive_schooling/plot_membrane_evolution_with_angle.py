import numpy as np
import matplotlib.pyplot as plt

from run_sim_files.zebrafish.hybrid_position_control.fictive_schooling.interpolate_experimental_angles import get_scaled_signal
# from run_sim_files.zebrafish.hybrid_position_control.fictive_schooling.plot_membrane_evolution_with_angle  import plot_membrane_and_angle_evolution

def _plot_module(
    statemon_t,
    statemon_v,
    times_angles,
    joint_angles,
    module,
    seg_ind,
    ner_ind    = 0,
    xlimits    = None,
    plot_label = ''
):
    ''' Plot the membrane potential evolution. '''

    if xlimits is None:
        xlimits = [0, statemon_t[-1]]

    ind_left  = module.indices_pools_sides[0][seg_ind][ner_ind]
    ind_right = module.indices_pools_sides[1][seg_ind][ner_ind]

    v_left  = statemon_v[:, ind_left]  * 1e3
    v_right = statemon_v[:, ind_right] * 1e3

    v_mean = np.mean([v_left, v_right])

    figname = f'Membrane evolution {module.name} {plot_label}'
    figname = figname.replace(' ', '_').replace('.', '_')

    fig = plt.figure(figname)
    plt.plot(
        statemon_t,
        v_left - v_mean,
        label     = 'Left',
        linewidth = 1.0,
    )
    plt.plot(
        statemon_t,
        v_right - v_mean,
        label     = 'Right',
        linewidth = 1.0,
    )

    plt.plot(
        times_angles,
        joint_angles,
        label     = 'Angle',
        linewidth = 2.0,
        color     = 'black',
    )

    plt.xlim(xlimits)
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane potential (mV)')
    plt.title(figname)
    plt.legend()
    plt.tight_layout()

    return fig

def plot_membrane_and_angle_evolution(
    freq_scaling,
    network_modules,
    statemon_dict,
    seg_ind = 5,
    ner_ind = 0,
):
    ''' Plot membrane and angle evolution. '''

    # freq_scaling = 0.3
    # network_modules = self.params.topology.network_modules
    # statemon_dict = self.statemon_dict

    statemon_t = statemon_dict['t']
    statemon_v = statemon_dict['v']

    timestep = statemon_t[1] - statemon_t[0]
    duration = statemon_t[-1]

    stim_duration = 0.7375 / freq_scaling
    stim_onset    = (duration - stim_duration) / 2
    stim_offset   = stim_onset + stim_duration

    joint_angles, times_angles = get_scaled_signal(
        timestep       = timestep,
        total_duration = duration,
        freq_scaling   = freq_scaling,
    )

    joint_angles = np.rad2deg(joint_angles)

    target_modules = [
        network_modules['cpg']['axial']['ex'],
        network_modules['mn']['axial']['mn'],
    ]

    fig_list = []
    for module in target_modules:

        for limits in [[0, duration], [stim_onset, stim_offset]]:

            fig = _plot_module(
                statemon_t,
                statemon_v,
                times_angles,
                joint_angles,
                module,
                seg_ind    = seg_ind,
                ner_ind    = ner_ind,
                xlimits    = limits,
                plot_label = f'{limits[0]:.2f} - {limits[1]:.2f} s',
            )

            fig_list.append(fig)

    # Save figures
    for fig in fig_list:
        fig.savefig(f'{fig._label}.pdf')

    plt.show(block=False)
    plt.pause(0.001)
    input("Enter to close plots and continue...")
    plt.close('all')