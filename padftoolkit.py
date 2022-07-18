import matplotlib.pyplot as plt
import numpy as np
import math as m
import os
import glob
import array

plt.rcParams['axes.linewidth'] = 0.2  # set the value globally
plt.rcParams["font.family"] = "Arial"


def gamma_scale(disp, gamma):
    disp = disp / gamma
    return disp


class ExpPADF:
    """
    Handler for Experimental PADF plots based around *.dbin files
    """

    def __init__(self, runnum='', group='', dir='', tag='', root='', nr=1, nq=1, nth=1):
        """
        File IO variables
        """

        self.tag = tag
        self.out_tag = tag
        self.group = group
        self.runnum = runnum
        self.dir = dir
        self.data_type = 'dbin'

        self.root = root
        self.out_root = self.root

        self.padf_volume = np.array([])  # Central analysis object
        self.corr_volume = np.array([])  # Central analysis object

        """
        PADF parameters
        """
        self.nr = nr
        self.nq = nq
        self.nth = nth
        self.rscale = 0.25
        self.pix = 128
        self.r_max = 25.0
        self.q_max = 10.0
        self.r_plot_lim = 0.1
        self.q_plot_lim = 0.1

        """
        Imaging parameters
        """
        self.aspect = 6.5
        self.gamma = 0.1
        self.cmap = 'viridis'
        self.gnuplot_mode = False

        """
        Load up the arrays
        """

    def db_in(self, vol_type):
        if 'padf' in vol_type:
            print(f"<db_in> Loading: {self.root}{self.tag}_padf2_padf.dbin")
            self.padf_volume = self.read_dbin(f'{self.root}{self.tag}_padf2_padf.dbin',
                                              nd=self.nr,
                                              nth=self.nth)
        if 'corr' in vol_type:
            print(f"<db_in> Loading: {self.root}{self.tag}_padfcorr_correlation_sum_maskcorrected.dbin")
            self.corr_volume = self.read_dbin(f'{self.root}{self.tag}_padfcorr_correlation_sum_maskcorrected.dbin',
                                              nd=128,
                                              nth=self.nth)
        if 'corr_trim' in vol_type:
            print(f"<db_in> Loading: {self.root}{self.tag}_padfcorr_correlation_sum_maskcorrected_trim.dbin")
            self.corr_volume = self.read_dbin(f'{self.root}{self.tag}_padfcorr_correlation_sum_maskcorrected_trim.dbin',
                                              nd=128,
                                              nth=self.nth)

    def corr_rescale(self, plane, gamma):
        disp = plane * 0.0
        ihigh = np.where(plane > 0)
        ilow = np.where(plane < 0)
        disp[ihigh] = np.abs(plane[ihigh]) ** gamma
        disp[ilow] = - np.abs(plane[ilow]) ** gamma
        return disp

    def read_dbin(self, path, swapbyteorder=0, nd=0, nth=0):
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

    def read_dbin_img(self, path, swapbyteorder=0, nd=0):
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
        print("<read_dbin> Target volume: ", nd * nd)
        sh_path = path.split('\\')[-1]
        print(f"<read_dbin> {sh_path} volume:  {output.size}")
        output = output.reshape(nd, nd)
        print(f"<read_dbin> {sh_path} shape : {output.shape}")
        output = np.array(output)
        return output

    def extract_rrprime_slice(self):
        print("extracting r=r' slice...")
        disp2 = np.zeros((self.nr, self.nth))
        for i in np.arange(self.nr):
            disp2[i, :] = self.volume[i, i, :]  # *r[i]*r[i]
        disp2 += - np.outer(np.average(disp2, 1), np.ones(self.nth))
        plt.imshow(disp2[:, :self.nth // 2], extent=[0, 180, 0, self.lim], origin='lower', aspect=2 // self.rscale)
        plt.show()
        print(disp2.shape)
        if self.gnuplot_mode:
            gamma_d = gamma_scale(disp2, np.max(disp2))
            np.savetxt(self.root + self.out_tag + '_reqr_gp.txt', gamma_d)
        return disp2

    def q_correlation_plot(self):
        corr = self.volume
        th = np.arange(self.nth) * 360.0 / float(self.nth)
        pix = self.nr  # * 1.0
        rmax = 1.88748  # nr #150.3
        lim = rmax * (pix / float(self.nr))
        gamma = 0.1
        print(pix)
        disp = self.corr_rescale(corr[:pix, :pix, 0], gamma)
        plt.imshow(disp, extent=[0, lim, 0, lim], origin='lower')
        plt.xlabel("$q$ ($\AA^{-1}$)")
        plt.ylabel("$q^\prime$ ($\AA^{-1}$)")
        plt.colorbar()
        plt.show()
        if self.gnuplot_mode:
            gamma_d = gamma_scale(disp, np.max(disp))
            np.savetxt(self.root + self.out_tag + '_qq_gp.txt', gamma_d)

    def corr_qeq_theta(self, plttitle='', clims=(1, 1), cmap='viridis', aspect=1,
                       show=False,
                       no_avg=False,
                       trim=False):
        if trim:
            self.db_in(vol_type='corr_trim')
        else:
            self.db_in(vol_type='corr')
        corr = self.corr_volume
        pix = self.nq  # * 1.0
        disp2 = np.zeros((self.nq, self.nth))
        thw = 0
        corr_av = np.zeros(self.nq)
        plt.figure()
        plt.title(f'{plttitle}')
        if no_avg:
            for i in np.arange(self.nq):
                for j in np.arange(self.nth):
                    disp2[i,j] = corr[i,i,j]
            disp2 = self.corr_rescale(disp2, self.gamma)
            plt.imshow(disp2[:int((self.q_plot_lim / self.q_max) * self.nq), :self.nth],
                       extent=[0, 360, 0, self.q_plot_lim],
                       origin='lower',
                       aspect=aspect,
                       cmap=cmap)
        else:
            for i in np.arange(self.nq):
                tot = 0
                for j in np.arange(self.nth):
                    tot = tot + corr[i, i, j]
                    average = tot / self.nth
                    corr_av[i] = average
            for i in np.arange(self.nq):
                for j in np.arange(self.nth):
                    disp2[i, j] = corr[i, i, j] - corr_av[i]
            disp2 = self.corr_rescale(disp2, self.gamma)
            # blur = True
            # if blur:
            #     disp2 = sp_gaussian_filter(disp2, sigma=1.5)
            plt.imshow(disp2[:int((self.q_plot_lim / self.q_max) * self.nq), :self.nth],
                       extent=[0, 360, 0, self.q_plot_lim],
                       origin='lower',
                       aspect=aspect,
                       cmap=cmap)
        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel(r'q (nm$^{-1}$)')
        plt.clim(clims[0], clims[1])
        # plt.show()
        display_arr = disp2[:pix, :self.nth - 2 * thw]
        print(display_arr.shape)

        return display_arr

    def corr_average_difference(self, q_max):
        corr = self.corr_volume
        thw = 0
        dispj = np.zeros((self.nr, self.nth))
        corr_av = np.zeros(self.nr)
        for i in np.arange(self.nr):
            tot = 0
            for j in np.arange(self.nth):
                tot = tot + corr[i, i, j]
                average = tot / self.nth
                corr_av[i] = average
        for i in np.arange(self.nr):
            for j in np.arange(self.nth):
                dispj[i, j] = corr[i, i, j] - corr_av[i]
        dispj = self.corr_rescale(dispj, self.gamma)
        if self.gnuplot_mode:
            gamma_d = gamma_scale(dispj, np.max(dispj))
            print("$1 factor: ", 360 / self.nth)
            print("$2 factor: ", (q_max / 1E8) / self.nr)
            np.savetxt(self.root + self.out_tag + '_qeqq_corr_av_diff_gp.txt', gamma_d)
        plt.imshow(dispj[30:self.pix - 10, :self.nth - 2 * thw], extent=[0, 360, 0, (q_max / 1E8)], origin='lower',
                   aspect=100)
        plt.show()

    def padf_reqr_theta(self, plttitle='', clims=(1, 1), c_dists='', cmap='', aspect=1, type='', show=False):
        print("extracting r-r' slice...")
        print(type)
        self.db_in(vol_type=type)
        disp = np.zeros((self.nr, self.nth))
        dr = np.zeros(self.nr)
        for i in np.arange(self.nr):
            disp[i, :] = self.padf_volume[i, i, :]  # *r[i]*r[i]
            dr[i] = self.padf_volume[i, i, 0]  # *r[i]*r[i]
        print(disp.shape)
        print(np.min(disp))
        print(np.max(disp))
        plt.figure()
        plt.title(f'{plttitle}')
        print((self.r_plot_lim / self.r_max) * self.nr)
        print(self.r_plot_lim)
        plt.imshow(disp[:int((self.r_plot_lim / self.r_max) * self.nr), :self.nth // 2],
                   extent=[0, 180, 0, self.r_plot_lim],
                   origin='lower',
                   aspect=aspect,
                   cmap=cmap)
        if self.gnuplot_mode:
            gamma_d = gamma_scale(disp, np.max(disp))
            print("$1 factor: ", 360 / self.nth)
            print("$2 factor: ", self.r_max / self.nr)
            np.savetxt(self.root + self.out_tag + '_reqr_padf_gp.txt', gamma_d)
        plt.ylabel("r = r' (nm)")
        plt.xlabel(r'$\theta$ (degrees)')
        plt.clim(clims[0], clims[1])
        plt.colorbar()
        plt.tight_layout()
        if len(c_dists) > 0:
            th_range = np.arange(0, 2 * np.pi, 0.01)
            for rbc in c_dists:
                f_th = (rbc) / (2 * np.sin(th_range / 2))
                plt.plot(np.rad2deg(th_range), f_th, label=f'{rbc} nm')
            plt.legend()
        plt.xlim(0, 180)
        plt.ylim(0, self.r_plot_lim)
        if show:
            plt.show()
        return disp

    def polar_slice(self, pie_slice):
        """
        Plots a 2D slice in polar coords.
        Needs updating with nice labels etc.
        """
        ax1 = plt.subplot(projection="polar")
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        x = np.linspace(0, pie_slice.shape[0], pie_slice.shape[0])
        if self.data_type == 'dbin':
            y = np.linspace(0, m.pi, pie_slice.shape[-1] // 2)
            ax1.pcolormesh(y, x, pie_slice[:, :pie_slice.shape[-1] // 2], shading='auto', cmap=self.cmap)
        else:
            y = np.linspace(0, m.pi, pie_slice.shape[-1])
            ax1.pcolormesh(y, x, pie_slice[:, :pie_slice.shape[-1]], shading='auto', cmap=self.cmap)
        plt.tight_layout()

    def polar_q_corr(self, pie_slice):
        """
        Plots a 2D slice in polar coords.
        Needs updating with nice labels etc.
        """
        ax1 = plt.subplot(projection="polar")
        ax1.set_thetamin(0)
        ax1.set_thetamax(360)
        x = np.linspace(0, pie_slice.shape[0], pie_slice.shape[0])
        if self.data_type == 'dbin':
            y = np.linspace(0, m.pi, pie_slice.shape[-1] // 2)
            ax1.pcolormesh(y, x, pie_slice[:, :pie_slice.shape[-1] // 2], shading='auto', cmap=self.cmap)
        else:
            y = np.linspace(0, 2 * m.pi, pie_slice.shape[-1])
            ax1.pcolormesh(y, x, pie_slice[:, :pie_slice.shape[-1]], shading='auto', cmap=self.cmap)
        plt.tight_layout()
        plt.show()

    def line_profile_plot(self, target_d=[], d_tol=0.0, show=False, filter=True,
                          dump=False, arr=[], array_tag='', nd=1, d_plot_lim=0):
        plt.figure()
        plt.xlabel(r'$\theta$ / $^\circ$')
        plt.ylabel(r'$\Theta(r = r^\prime)$')
        arr = arr
        of = array_tag
        print(f'{arr.shape}  arr shape')
        th_range = np.linspace(start=0, stop=360, num=self.nth)
        for roi in target_d:
            pix_per_r = nd / d_plot_lim
            target_index = int(roi * pix_per_r)
            if d_tol > 0.0:
                target_min_index = int((roi - d_tol) * pix_per_r)
                target_max_index = int((roi + d_tol) * pix_per_r)
                slicer = np.sum(arr[target_min_index:target_max_index, :], axis=0)

                plt.plot(slicer, label=str(f'{roi}({d_tol})'),
                         extent=[0, 360])
                if dump:
                    np.savetxt(f'{self.root}{self.project}{self.tag}_lineplot_{of}_r{roi}.txt', slicer)

            else:
                slicer = arr[target_index, :]
                print(f'slicer.shape : {slicer.shape}')
                plt.plot(th_range, slicer, label=str(roi))
                if dump:
                    np.savetxt(f'{self.root}{self.project}{self.tag}_lineplot_{of}_r{roi}.txt', slicer)
        plt.legend()
        if show:
            plt.show()

    def corvol_reqr_theta(self):
        print("extracting r-r' slice...")
        disp = np.zeros((self.nr, self.nth))
        dr = np.zeros(self.nr)
        ir = int(self.r1 * (self.nr / float(self.rmax)))
        for i in np.arange(self.nr):
            disp[i, :] = self.volume[ir, i, :]  # *r[i]*r[i]
            dr[i] = self.volume[i, i, 0]  # *r[i]*r[i]
        disp += - np.outer(np.average(disp, 1), np.ones(self.nth))
        print(disp.shape)
        plt.imshow(disp[:self.pix, :self.nth], extent=[0, 180, 0, self.r_max], origin='lower',
                   aspect=self.aspect,
                   cmap=self.cmap)
        if self.gnuplot_mode:
            gamma_d = gamma_scale(disp, np.max(disp))
            print("$1 factor: ", 360 / self.nth)
            print("$2 factor: ", self.rmax / self.nr)
            np.savetxt(self.root + self.out_tag + '_reqr_padf_gp.txt', gamma_d)
        plt.ylabel("r = r' (nm)")
        plt.xlabel(r'$\theta$ (degrees)')
        plt.clim(np.min(disp) * 0.5, np.max(disp) * 0.5)
        plt.ylim((0, self.r_max))
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        return disp

    def summary_plots(self, reqrpadf=False, qeqqcorr=False,
                      plttitle='', clims=(1, 1), c_dists='', cmap='viridis',
                      aspect=20):
        if qeqqcorr:
            self.corr_qeq_theta(plttitle=plttitle, clims=clims, cmap=cmap, aspect=aspect)
        if reqrpadf:
            self.padf_reqr_theta(plttitle=plttitle, clims=clims, c_dists=c_dists, cmap=cmap, aspect=aspect, type='padf')


    def theta_slices(self, theta_vals=[], arr='', clims=(1, 1), cmap='viridis', show=False):
        print(f'<theta_slices> Generating {len(theta_vals)} theta slices...')
        if arr == 'padf':
            volume = self.padf_volume
        elif arr == 'corr':
            volume = self.corr_volume
        else:
            print('<theta_slices> No arr tag supplied')
        theta_indexes = []
        print(f'<theta_slices> {volume.shape}')
        for theta in theta_vals:
            # for ioi in [0, 200, 401, ]:
            ioi = int((theta / 180.0) * (self.nth // 2))
            print(f'<theta_slices> {theta}° - {ioi} index')
            theta_indexes.append(ioi)
            disp = volume[:, :, ioi]
            plt.figure()
            plt.title(f'{self.runnum} - theta = {theta}')
            plt.imshow(
                disp[:int((self.r_plot_lim / self.r_max) * self.nr),
                :int((self.r_plot_lim / self.r_max) * self.nr)],
                extent=[0, self.r_plot_lim, 0, self.r_plot_lim],
                origin='lower',
                cmap=cmap
            )
            plt.clim(clims[0], clims[1])
            plt.colorbar()
            plt.tight_layout()
            plt.xlabel("r (nm)")
            plt.ylabel("r' (nm)")
        if show:
            plt.show()

    def diff_inspect(self, clims=(0,1), vol_type='', show=False):
        if vol_type == 'diff':
            diffdiff_list = glob.glob(f'{self.root}*diffdiffraction*.dbin')
            for diff in diffdiff_list:
                img_data = self.read_dbin_img(diff, nd=256)
                plt.figure()
                plt.imshow(img_data)
                plt.title(diff)
                plt.clim(clims[0], clims[1])
        elif vol_type == 'corr':
            diffdiff_list = glob.glob(f'{self.root}*diffdiffraction*.dbin')
            for diff in diffdiff_list:
                corr = self.read_dbin(diff, nd=self.nq, nth=self.nth)
                plt.figure()
                disp = np.zeros((self.nq, self.nth))
                for i in np.arange(self.nq):
                    for j in np.arange(self.nth):
                        disp[i, j] = corr[i, i, j]
                plt.title(diff)
                plt.imshow(disp, origin='lower')
                plt.clim(clims[0], clims[1])
        else:
            print(f'<diff_inspect> WARNING: no vol_type ["diff"/"corr"] given!')
        if show:
            plt.show()