import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import array

# Global plotting params
# Feel free to import matplotlib style sheet instead
plt.rcParams['axes.linewidth'] = 0.2  # set the value globally
plt.rcParams["font.family"] = "Arial"


def normalize_array(data):
    """
    Normalize an array between -1, and 1.
    :param data: array to be normalized
    :return: array normalized between -1 and 1
    """
    return -1 + ((data - np.min(data) * 2.0) / (np.max(data) - np.min(data)))


class ConfigReader:
    """
    Basic object to read padf config files. Should be deprecated once py3padf moves to standard config files.
    Takes a file_path variable pointing to {tag}_padf_config.txt
    """

    def __init__(self, file_path: str = ''):
        self.config_file = file_path
        self.param_dict = {
            'correlationfile': {'val': '', 'param_type': str},
            'outpath': {'val': '', 'param_type': str},
            'tag': {'val': '', 'param_type': str},
            'wavelength': {'val': 0.0, 'param_type': float},
            'nthq': {'val': 1, 'param_type': int},
            'nq': {'val': 1, 'param_type': int},
            'nr': {'val': 1, 'param_type': int},
            'nl': {'val': 1, 'param_type': int},
            'nlmin': {'val': 1, 'param_type': int},
            'qmax': {'val': 0.0, 'param_type': float}, 'rmax': {'val': 0.0, 'param_type': float}
        }

    def read_padf_config(self):
        with open(self.config_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                spline = line.split()
                self.param_dict[spline[0]]['val'] = self.param_dict[spline[0]]['param_type'](spline[-1])
            # print(self.param_dict)
            [print(key, param) for key, param in self.param_dict.items()]


class TestData:
    """
    Module to run basic tests during writing and debugging
    """

    def __init__(self):
        self.cfg = ConfigReader

    def read_config(self):
        cfg = ConfigReader(file_path='test_data\\TEST_padf_config.txt')
        cfg.read_padf_config()

    def run_test(self):
        print('Running test, attempting to read param file and load dictionary...')
        padfplotter = PadfPlotter(root='test_data\\', tag='TEST', read_config=True)
        padfplotter.plot_d_eq_d(key='padf', show=False, save=False, d_plot_lim=4.0e-09,
                                dists=((1, 2, 3, 4, 5,)), clims=(-5e52, 5e52), dist_label_type='annotate')
        # padfplotter.plot_d_eq_d(key='corr', show=False, d_plot_lim=3701500000)
        plt.show()


class PadfPlotter:
    """
    Handler for Experimental PADF plots using py3padf output
    """

    def __init__(self, root: str = '', tag: str = '',
                 nr: int = 0, nq: int = 0, nth: int = 0,
                 read_config=True):
        # File I/O variables
        self.root = root
        self.tag = tag

        # Volume objects & parameters
        self.padf_volume = np.array([])
        self.corr_volume = np.array([])
        self.nr = nr
        self.nq = nq
        self.nth = nth
        self.rscale = 0.25
        self.pix = 0  # Is this needed?
        self.rmax = 0.0
        self.qmax = 0.0
        self.r_plot_lim = 0.0  # r limit to plot to
        self.q_plot_lim = 0.0  # q limit to plot to

        """
        Display parameters
        """
        self.cmap = 'viridis'
        self.gnuplot_mode = False

        if read_config:
            cfg = ConfigReader(file_path=f'{self.root}\\{self.tag}_padf_config.txt')
            cfg.read_padf_config()
            # Pass config values to this object
            self.nth = cfg.param_dict['nthq']['val']
            self.nq = cfg.param_dict['nq']['val']
            self.nr = cfg.param_dict['nr']['val']
            self.qmax = cfg.param_dict['qmax']['val']
            self.rmax = cfg.param_dict['rmax']['val']
        else:
            print(f'<padf_toolkit.PadfPlotter> I am not attempting to read a config file. Please set PADF parameters '
                  f'manually.')

        """
        This dictionary contains all the required parameters for each kind of plot. This allows the same functions
        to be applied to corrvol and padfvol plots. Can be extended for other data types if required.
        """
        self.plot_property_dict = {
            'padf': {'default_scaling': 1E-9, 'dist_label': "r = (nm)", 'deqd_label': "r = r' (nm)",
                     'volume': self.padf_volume, 'd_param': self.nr, 'd_plot_limit': self.r_plot_lim,
                     'dmax': self.rmax, 'theta_extent': 180, 'theta_lim': self.nth // 2,
                     'dbin_suffix': '_padf2_padf.dbin'},
            'corr': {'default_scaling': 1E9, 'dist_label': r"q (nm$^{-1}$)", 'deqd_label': "q = q' (nm$^{-1}$)",
                     'volume': self.corr_volume, 'd_param': self.nq, 'd_plot_limit': self.q_plot_lim,
                     'dmax': self.qmax, 'theta_extent': 360, 'theta_lim': self.nth,
                     'dbin_suffix': '_padfcorr_correlation_sum_maskcorrected.dbin'
                     }
        }

    def plot_d_eq_d(self, title: str = '', d_plot_lim: float = 0.0, dists: tuple = (),
                    dist_line_color: str = 'white',
                    dist_label_type: str = 'legend',
                    key: str = 'padf', clims: tuple = (),
                    save: bool = False,
                    show: bool = False):
        """
        Plot the classic r(q) = r'(q') vs theta plot. Here we use a generic variable 'd' to refer to either q/r.
        :param title: string to furnish the plot with a title
        :param d_plot_lim: distance IN METERS that marks the maximum value of 'd' to plot to.
        :param dists: a tuple giving a list of characteristic distances to plot. If just plotting one write: (d,)
        :param dist_line_color: color of the characteristic distance line plots.
        :param dist_label_type: ['legend'/'annotate'] Chose the style of labelling for the characteristic plots.
        :param key: figure type, ['padf' / 'corr']
        :param clims: directly state the min and max color values for the imshow
        :param save: write image to png
        :param show: display the image. Turn off to stack multiple figures for inspection
        :return: primarily plots, but can also return the display slice.
        """
        print(f'<plot_d_eq_d> plot type {key}')
        print(f"<plot_d_eq_d> plotting {self.plot_property_dict[key]['deqd_label']}")
        # Set variables from dict and check to see what has been passed in and what needs defaulting
        nd = self.plot_property_dict[key]['d_param']
        dmax = (self.plot_property_dict[key]['dmax']) / self.plot_property_dict[key]['default_scaling']
        d_plot_lim = d_plot_lim / self.plot_property_dict[key]['default_scaling']
        theta_extent = self.plot_property_dict[key]['theta_extent']
        theta_lim = self.plot_property_dict[key]['theta_lim']
        # First we check if the volume has already been read in:
        if np.size(self.plot_property_dict[key]['volume']) == 0:
            # Volume is empty, go grab it. This is slow
            volume = self.read_dbin(path=f"{self.root}{self.tag}{self.plot_property_dict[key]['dbin_suffix']}",
                                    nd=nd,
                                    nth=int(self.nth))
        else:
            volume = self.plot_property_dict[key]['volume']  # Volume has non-zero size, and so we simply point to it
        # Create blank display arrays
        disp = np.zeros((nd, self.nth))
        disp_r = np.zeros(nd)
        # Fill with values
        for i in np.arange(nd):
            disp[i, :] = volume[i, i, :]
            disp_r[i] = volume[i, i, 0]

        # Create the figure
        plt.figure()
        plt.title(title)
        # disp = self.normalize_array(disp)
        plt.imshow(disp[:int((d_plot_lim / dmax) * nd), : theta_lim],
                   extent=[0, theta_extent, 0, d_plot_lim],
                   origin='lower',
                   # aspect=self.aspect,
                   aspect='auto',
                   cmap=self.cmap,
                   interpolation='none')
        plt.ylabel(self.plot_property_dict[key]['deqd_label'])
        plt.xlabel(r'$\theta$ (degrees)')
        # Set the clims, either manually or using default guesses based on np.min and max
        plt.clim(np.min(disp) * 0.1, np.max(disp) * 0.1) if not clims else plt.clim(clims[0], clims[1])
        # Characteristic distance plotter
        if dists:
            self.characteristic_distance_plotter(dists=dists,
                                                 dist_line_color=dist_line_color,
                                                 dist_label_type=dist_label_type)
        plt.xlim(0, theta_extent)
        plt.ylim(0, d_plot_lim)
        if save:
            plt.savefig(f'{self.root}{self.tag}_deqd_{key}.png')
            print(f'Figure saved to {self.root}{self.tag}_deqd_{key}.png')
        if show:
            plt.show()
        return disp[:int((d_plot_lim / dmax) * nd), : theta_lim]

    """
    Characteristic distance functions
    """

    def characteristic_distance_plotter(self, dists: tuple = (), key: str = 'padf',
                                        dist_line_color: str = 'white',
                                        dist_label_type: str = 'legend',
                                        dist_label_size: int = 12,
                                        dist_label_color: str = 'white',
                                        dist_mask: tuple = (155, 170)):
        """
        Function for handling the plotting of characterisic distances on d=d' slices.
        :param dists: a tuple giving a list of characteristic distances to plot. If just plotting one write: (d,)
        :param key: figure type, ['padf' / 'corr']
        :param dist_line_color: color of the characteristic distance line plots.
        :param dist_label_type:  ['legend'/'annotate'] Chose the style of labelling for the characteristic plots.
        :param dist_label_size: text size for labels
        :param dist_label_color: text color for labels
        :param dist_mask: tuple, with mask limits for the 'annotate' style.
        :return:
        """
        th_range = np.arange(0.01, np.deg2rad(self.plot_property_dict[key]['theta_extent']), 0.01)
        f_ths = []
        label_values = []
        dists = sorted(dists)
        for k, rbc in enumerate(sorted(dists)):
            f_th = rbc / (2 * np.sin(th_range / 2))
            f_ths.append(f_th)
            if dist_label_type == 'legend':
                plt.plot(np.rad2deg(th_range), f_th, label=f'{rbc} nm')
            elif dist_label_type == 'annotate':
                label_values.append(rbc / (2 * np.sin((0.9 * np.pi) / 2)))
                th_range_mask = np.ma.masked_inside(th_range, np.deg2rad(dist_mask[0]), np.deg2rad(dist_mask[1]))
                f_th_mask = f_th * np.invert(th_range_mask.mask)
                plt.plot(np.rad2deg(th_range_mask), f_th_mask,
                         color=dist_line_color,
                         linewidth=0.2)
                ypos = label_values[k]
                plt.text((0.87 * 180), ypos * 0.95, f'{dists[k]} nm', fontsize=dist_label_size, color=dist_label_color)
        if dist_label_type == 'legend':
            plt.legend()

    """
    Utility Functions
    """

    def read_dbin(self, path, swapbyteorder=0, nd=0, nth=0):
        """
        Read in PADF/corr volume saved as a dbin
        :param path: path to dbin volume
        :param swapbyteorder: flag to swap byteorder (depracated?)
        :param nd: number of distance pixels (nr/nq)
        :param nth: number of angular pixels
        :return: reshaped array containing dbin data.
        """
        size = os.path.getsize(path)
        b = array.array('d')
        f = open(path, "rb")
        b.fromfile(f, size // 8)
        f.close()
        elle = b.tolist()
        output = np.array(elle)
        if swapbyteorder == 1:
            output = output.newbyteorder()
        print("<read_dbin> Reshaping...")
        print("<read_dbin> Target volume: ", nd * nd * nth)
        sh_path = path.split('\\')[-1]
        print(f"<read_dbin> {sh_path} volume:  {output.size}")
        output = output.reshape(nd, nd, nth)
        print(f"<read_dbin> {sh_path} shape : {output.shape}")
        output = np.array(output)
        return output


if __name__ == '__main__':
    test = TestData()
    # test.read_config()
    test.run_test()
