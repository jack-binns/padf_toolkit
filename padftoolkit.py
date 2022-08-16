import array
import os
import timeit

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Global plotting params
# Feel free to import matplotlib style sheet instead
plt.rcParams['axes.linewidth'] = 0.5  # set the value globally
plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.edgecolor'] = 'black'
sns.set_style('ticks')
cmap = sns.color_palette("crest", as_cmap=True)


def normalize_array(data):
    """
    Normalize an array between -1, and 1.
    :param data: array to be normalized
    :return: array normalized between -1 and 1
    """
    return -1 + ((data - np.min(data) * 2.0) / (np.max(data) - np.min(data)))


class ConfigReader:
    """
    Basic object to read padf config files. Should be deprecated once py3padf moves to standard config files which can
    be read with configparser.
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
        padfplotter.plot_d_eq_d(key='padf', show=False, save=False, d_plot_lim=8.0e-09,
                                dists=((1, 2, 3, 4, 5,)), clims=(-5e52, 5e52), dist_label_type='annotate')
        # padfplotter.plot_d_eq_d(key='padf', show=False, save=False, d_plot_lim=8.0e-09,
        #                         dists=((1, 2, 3, 4, 5, 6, 7, 8)), clims=(-5e52, 5e52), dist_label_type='annotate')
        # padfplotter.plot_d_eq_d(key='corr', show=False, d_plot_lim=3701500000)
        # padfplotter.polar_slice(target_r=2.3e-09)
        # padfplotter.polar_slice(target_r=4.8e-09, title=r'$r = 4.8$ nm', clims=(-5E52, 5E52), d_plot_lim=8.0)
        padfplotter.line_section(key='padf', target_r=(4.8e-09, 2.3e-09, 6.0e-09))
        # self.run_file_io_speedtest()
        plt.show()

    def run_file_io_speedtest(self):
        t_dbin = timeit.repeat(lambda: self.test_dbin_read(), number=5, repeat=3)
        t_npy = timeit.repeat(lambda: self.test_npy_read(), number=5, repeat=3)
        print(f" npy reading: {np.mean(np.array(t_npy))} s")
        print(f" dbin reading: {np.mean(np.array(t_dbin))} s")

    def test_dbin_read(self):
        path = 'test_data\\TEST_padf2_padf.dbin'
        size = os.path.getsize(path)
        b = array.array('d')
        f = open(path, "rb")
        b.fromfile(f, size // 8)
        f.close()
        elle = b.tolist()
        output = np.array(elle)
        print("<read_dbin> Reshaping...")
        print("<read_dbin> Target volume: ", 512 * 512 * 402)
        sh_path = path.split('\\')[-1]
        print(f"<read_dbin> {sh_path} volume:  {output.size}")
        output = output.reshape((512, 512, 402))  #
        print(f"<read_dbin> {sh_path} shape : {output.shape}")
        output = np.array(output)
        return output

    def test_npy_read(self):
        output = np.load(f"test_data\\TEST_padf2_padf.npy")
        return output


class PadfPlotter:
    """
    Handler for Experimental PADF plots using py3padf output
    """

    def __init__(self, root: str = '', tag: str = '',
                 nr: int = 0, nq: int = 0, nth: int = 0,
                 rmax: float = 0.0, qmax: float = 0.0,
                 r_plot_lim: float = 0.0, q_plot_lim: float = 0.0,
                 read_config: bool = True, npy_bkup_flag: bool = True):
        # File I/O variables
        self.root = root
        self.tag = tag
        self.npy_bkup_flag = npy_bkup_flag

        # Volume objects & parameters
        self.padf_volume = np.array([])
        self.padf_reqr = None
        self.corr_volume = np.array([])
        self.corr_qeqq = None
        self.nr = nr
        self.nq = nq
        self.nth = nth
        self.rscale = 0.25
        self.pix = 0  # Is this needed?
        self.rmax = rmax
        self.qmax = qmax
        self.r_plot_lim = r_plot_lim  # r limit to plot to
        self.q_plot_lim = q_plot_lim  # q limit to plot to

        """
        Display parameters
        """
        self.cmap = 'viridis'
        self.gnuplot_mode = False

        if read_config:
            cfg = ConfigReader(file_path=f'{self.root}\\{self.tag}_padf_config.txt')
            cfg.read_padf_config()
            # Pass config values to this object
            self.nth = int(cfg.param_dict['nthq']['val'])
            self.nq = int(cfg.param_dict['nq']['val'])
            self.nr = int(cfg.param_dict['nr']['val'])
            self.qmax = float(cfg.param_dict['qmax']['val'])
            self.rmax = float(cfg.param_dict['rmax']['val'])
        else:
            print(f'<padf_toolkit.PadfPlotter> I am not attempting to read a config file. Please set PADF parameters '
                  f'manually.')

        """
        This dictionary contains all the required parameters for each kind of plot. This allows the same functions
        to be applied to corrvol and padfvol plots. Can be extended for other data types if required.
        """
        self.plt_props = {
            'padf': {'default_scaling': 1E-9,
                     'dist_label': r"$r$ (nm)",
                     'deqd_label': r"$r = r^\prime$ (nm)",
                     'dist_unit': "nm",
                     'volume': self.padf_volume,
                     'd_param': self.nr,
                     'd_plot_limit': self.r_plot_lim,
                     'dmax': self.rmax,
                     'theta_extent': 180,
                     'theta_limit': self.nth // 2,
                     'dbin_suffix': '_padf2_padf'},

            'corr': {'default_scaling': 1E9,
                     'dist_label': r"$q$ (nm$^{-1}$)",
                     'deqd_label': r"$q = q^\prime$ (nm$^{-1}$)",
                     'dist_unit': r"nm$^{-1}$",
                     'volume': self.corr_volume,
                     'd_param': self.nq,
                     'd_plot_limit': self.q_plot_lim,
                     'dmax': self.qmax,
                     'theta_extent': 360,
                     'theta_limit': self.nth,
                     'dbin_suffix': '_padfcorr_correlation_sum_maskcorrected'
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
        print(f"<plot_d_eq_d> plotting {self.plt_props[key]['deqd_label']}")
        # Set variables from dict and check to see what has been passed in and what needs defaulting
        nd = self.plt_props[key]['d_param']
        dmax = (self.plt_props[key]['dmax']) / self.plt_props[key]['default_scaling']
        d_plot_lim = d_plot_lim / self.plt_props[key]['default_scaling']
        theta_extent = self.plt_props[key]['theta_extent']
        theta_lim = self.plt_props[key]['theta_limit']
        # Volume input
        volume = self.file_read_handler(key=key)
        # Create blank display arrays
        disp = np.zeros((nd, self.nth))
        disp_th_zero_int = np.zeros(nd)
        # Fill with values
        for i in np.arange(nd):
            disp[i, :] = volume[i, i, :]
            disp_th_zero_int[i] = volume[i, i, 0]
        # Create the image figure
        plt.figure()
        plt.title(title)
        plt.imshow(disp[:int((d_plot_lim / dmax) * nd), : theta_lim],
                   extent=[0, theta_extent, 0, d_plot_lim],
                   origin='lower',
                   # aspect=self.aspect,
                   aspect='auto',
                   cmap=self.cmap,
                   interpolation='none')
        plt.ylabel(self.plt_props[key]['deqd_label'])
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
        # If the key is 'corr' we want to also subtract the angular average
        if key == 'corr':
            self.corr_average_subtraction(volume=volume, d_plot_lim=d_plot_lim)

        # PLot the theta_0 line
        plt.figure()
        plt.xlabel(self.plt_props[key]['dist_label'])
        plt.ylabel(r'Correlation intensity at $\theta=0°$')
        plt.xlim(0, self.plt_props[key]['dmax'])
        plt.plot(np.linspace(start=0,
                             stop=self.plt_props[key]['dmax'],
                             num=self.plt_props[key]['d_param']),
                 disp_th_zero_int, color=cmap(1.0))

        # Save and show checks
        if save:
            plt.savefig(f'{self.root}{self.tag}_deqd_{key}.png')
            print(f'Figure saved to {self.root}{self.tag}_deqd_{key}.png')
        if show:
            plt.show()
        if self.padf_reqr is None and key == 'padf':
            self.padf_reqr = disp[:int((d_plot_lim / dmax) * nd), : theta_lim]
            print(f'created padf r=r slice {self.padf_reqr.shape}')
        elif self.corr_qeqq is None and key == 'corr':
            self.corr_qeqq = disp[:int((d_plot_lim / dmax) * nd), : theta_lim]
            print(f'created corr q=q slice {self.corr_qeqq.shape}')
        return disp[:int((d_plot_lim / dmax) * nd), : theta_lim]

    def corr_average_subtraction(self, volume=np.array([]), d_plot_lim: float = 0.0, show=False):
        """
        Internal function to subtract the angular mean from the correlation volume plots
        :param volume: numpy array, will typically be the correlation volume but the function is flexible
        :param d_plot_lim: maximum plotting distance (typically q)
        :param show: display flag
        :return: disp: the mean-corrected correlation volume.
        """
        nq = self.plt_props['corr']['d_param']
        dmax = (self.plt_props['corr']['dmax']) / (self.plt_props['corr']['default_scaling'])
        theta_lim = self.plt_props['corr']['theta_limit']
        theta_extent = self.plt_props['corr']['theta_extent']
        disp = np.zeros((nq, self.nth))
        for i in np.arange(nq):
            for j in np.arange(self.nth):
                disp[i, j] = volume[i, i, j]
        theta_avg = np.mean(disp, axis=1)
        print(f'{theta_avg.shape=}')
        disp -= theta_avg[:, None]
        # Figure details
        plt.figure()
        plt.imshow(disp[:int((d_plot_lim / dmax) * nq), : theta_lim],
                   extent=[0, theta_extent, 0, d_plot_lim],
                   origin='lower',
                   aspect='auto',
                   cmap=self.cmap,
                   interpolation='none')
        plt.ylabel(f"Angular-mean-subtracted {self.plt_props['corr']['deqd_label']}")
        plt.xlabel(r'$\theta$ (degrees)')
        plt.clim(np.min(disp) * 0.1, np.max(disp) * 0.1)
        plt.xlim(0, theta_extent)
        plt.ylim(0, d_plot_lim)
        if show:
            plt.show()
        return disp

    def polar_slice(self, target_r: float = 0.0, title: str = '', clims: tuple = (),
                    d_plot_lim: float = None, show=False):
        """
        Polar slice will plot an r vs theta polar semicircle for r= target_r
        :param show: display flag, False by default
        :param target_r: fixed r value for which r' and theta are displayed
        :param title: title of the plot
        :param clims: colour bar limits
        :param d_plot_lim: r' limit for display, here unit scaling is applied based on the plt_props dictionary
        :return: passed axis object back.
        """
        volume = self.file_read_handler(key='padf')
        unit_scale = self.plt_props['padf']['default_scaling']
        target_r /= unit_scale
        rmax = self.rmax / unit_scale
        r_ioi = int((target_r / rmax) * self.nr)
        print(f'<polar_slice> target r :: {target_r} ==> {r_ioi}')
        r_theta_slice = volume[r_ioi, :, :]
        print(f'<polar_slice> r_theta_slice shape: {r_theta_slice.shape}')
        plt.figure()
        ax1 = plt.subplot(projection="polar")
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        plt.xlabel(self.plt_props['padf']['dist_label'])
        plt.ylabel(r'$\theta$ (degrees)', rotation='horizontal')
        ax1.xaxis.set_label_coords(0.75, 0.15)
        ax1.yaxis.set_label_coords(0.5, 0.85)
        plt.title(title)
        x = np.linspace(0, rmax, r_theta_slice.shape[0])
        y = np.linspace(0, np.pi, r_theta_slice.shape[-1] // 2)
        if not clims:
            clims = (np.min(r_theta_slice) * 0.1, np.max(r_theta_slice) * 0.1)
        else:
            clims = (clims[0], clims[1])
        if d_plot_lim:
            ax1.set_ylim([0, d_plot_lim])
        ax1.pcolormesh(y, x, r_theta_slice[:, :r_theta_slice.shape[-1] // 2],
                       shading='auto',
                       cmap=self.cmap,
                       vmin=clims[0], vmax=clims[1])
        plt.tight_layout()
        if show:
            plt.show()
        return ax1

    def line_section(self, key: str = '', target_r: tuple = (), show=False, title=''):
        if not target_r:
            print('<line_section> no target r values provided, check param target_r=(r1, r2,...,rn)')
        if key == 'padf':
            deqd = self.padf_reqr
        elif key == 'corr':
            deqd = self.corr_qeqq
        else:
            print('<line_section> provide key = ["padf"/"corr"]')
            deqd = None
        # Grab relevant values from the plt_props dict
        dmax = self.plt_props[key]['dmax']
        d_param = self.plt_props[key]['d_param']
        unit_scale = self.plt_props[key]['default_scaling']
        # Create list of matching indexes from which we will extract line sections
        target_r_indexes = []
        [target_r_indexes.append(int((ta_rt / dmax) * d_param)) for ta_rt in sorted(target_r)]
        # Beautify the target_r values for plotting
        target_r_labels = []
        [target_r_labels.append(ta_rt / unit_scale) for ta_rt in sorted(target_r)]
        # print(f'{target_r_indexes=}')
        # Create the figure object
        plt.figure()
        plt.title(title)
        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel('Correlation Intensity (arb. units)')
        theta_range = np.linspace(start=0, stop=self.plt_props[key]['theta_extent'],
                                  num=self.plt_props[key]['theta_limit'])
        for k, ta_rt in enumerate(target_r_indexes):
            plt.plot(theta_range, deqd[ta_rt, :],
                     label=f"{target_r_labels[k]} {self.plt_props[key]['dist_unit']}",
                     color=cmap(k / (len(target_r_indexes) - 1)))
        plt.xlim((0, self.plt_props[key]['theta_extent']))
        plt.legend()
        if show:
            plt.show()

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
        th_range = np.arange(0.01, np.deg2rad(self.plt_props[key]['theta_extent']), 0.01)
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
        return f_ths

    """
    Utility Functions
    """

    def file_read_handler(self, key):
        """
        Handles the reading, and backup to npy, of corr and padf volume files.
        Can deal with legacy dbin's as well as current npy formats. By default
        we also write out a npy file if it's not done already. This can be controlled
        with the self.npy_bkup_flag.
        :param key: ['padf'/'corr'] determines which volume type is read in
        :return: returns the volume
        """
        # First we check if the volume has already been read in:
        volume = None
        if np.size(self.plt_props[key]['volume']) != 0:
            print('<file_read_handler> Volume already read in, passing back')
            volume = self.plt_props[key]['volume']  # Volume has non-zero size, and so we simply point to it
            return volume
        if np.size(self.plt_props[key]['volume']) == 0:
            # Volume is empty, let's grab it
            print('<file_read_handler> No volume in memory, reading from disk')
            # First let's check if a npy array has been written out and can be read
            try:
                print(
                    f"<file_read_handler> Looking for {self.root}{self.tag}{self.plt_props[key]['dbin_suffix']}.npy")
                volume = np.load(f"{self.root}{self.tag}{self.plt_props[key]['dbin_suffix']}.npy")
                self.plt_props[key]['volume'] = volume
            except IOError:
                print('<file_read_handler> No npy file found...reading dbin instead ')
                volume = self.read_dbin(path=f"{self.root}{self.tag}{self.plt_props[key]['dbin_suffix']}.dbin",
                                        nd=self.plt_props[key]['d_param'],
                                        nth=int(self.nth))
                self.plt_props[key]['volume'] = volume
                if self.npy_bkup_flag:
                    np.save(f"{self.root}{self.tag}{self.plt_props[key]['dbin_suffix']}.npy", volume)
                    print(
                        f"<file_read_handler> I wrote a back up to: {self.root}{self.tag}{self.plt_props[key]['dbin_suffix']}.npy")
            finally:
                return volume

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
