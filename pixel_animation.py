'''
========================
Pixel animation
========================

Create an animation of the physical information content of each pixel.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plotpar = {'axes.labelsize': 35,
           'font.size': 20,
           'legend.fontsize': 18,
           'xtick.labelsize': 25,
           'ytick.labelsize': 25,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def load_real_ifft(df, Nstar):
    reals, flags = [], []
    for i, kepid in enumerate(df.kepid.values[:Nstar]):
        print("loading ifft", i, "of", len(df.kepid.values[:Nstar]))

        ifft_file_name = "data/ifft_real/{}_ifft.csv".format(kepid)
        if os.path.exists(ifft_file_name):
            ifft_df = pd.read_csv(ifft_file_name)
            real, imag = ifft_df.real, ifft_df.imag
            flag = 1
        else:
            real, imag = [np.zeros(100) for i in range(2)]
            flag = 0

        reals.append(np.sqrt(np.array(real)**2))
        flags.append(flag)
    return reals, flags


def make_animation(df, reals, flags, param, n_pixels=100, Nstar=100,
                   linefit=False):
    """
    param should be string, like "teff" or "logg".
    """
    for n in range(n_pixels):
        print("pixel", n, "of", n_pixels)
        # Get array of pixel values at the nth pixel.
        pixel = np.array([reals[i][n] for i in range(len(reals))])

        # Remove stars where no psd was downloaded.
        m = np.array(flags) > 0

        plt.clf()
        # fig, ax = plt.subplots(figsize=(20, 10))
        plt.figure(figsize=(20,10))

        # Plot the first 100 pixels of the psd for each star.
        plt.subplot(1, 2, 1)
        for i, kepid in enumerate(df.kepid.values[:Nstar]):
            plt.plot(np.log10(reals[i][:n_pixels]), "k-", alpha=.1)
            plt.plot(n, np.log10(reals[i][:n_pixels][n]), ".", color="r",
                     ms=5)
        plt.xlim(0, n_pixels)
        plt.ylim(-3, 3)
        # plt.xlabel("$\\Delta t [\mathrm{Days}]$")
        plt.ylabel("$\log_{10}(|\Re(ifft)|)$")

        # Plot the pixel value vs effective temperature.
        plt.subplot(1, 2, 2)
        plt.plot(df[param].values[:Nstar][m], np.log10(pixel[m]), ".", ms=10)

        if linefit:
            x = df[param].values[m]
            y = np.log10(pixel[m])
            AT = np.vstack((x, np.ones_like(x)))
            ATA = np.dot(AT, AT.T)
            m, c = np.linalg.solve(ATA, np.dot(AT, y))
            xplot = np.linspace(min(x), max(x), 100)
            plt.plot(xplot, m*xplot + c, ls="--", color=".5")

        if param == "teff":
            plt.xlabel = "$T_{\mathrm{eff}}~[K]$"
            plt.xlim(4500, 7000)
        elif param == "logg":
            plt.xlabel = "$\log(g)$"
        plt.ylim(-3, 3)
        plt.ylabel("${0}$".format(n))
        # plt.subplots_adjust(left=.15, bottom=.15, wspace=)
        plt.savefig("movie/pixel{0}_{1}".format(str(n).zfill(4), param))


def save_as_movie(param, framerate=10, quality=25):
    """
    Make the movie file
    """
    os.system("/Applications/ffmpeg -r {0} -f image2 -s 1920x1080 -i "\
            "movie/pixel%04d_{2}.png -vcodec libx264 -crf {1}  -pix_fmt "\
            "yuv420p {2}_movie.mp4".format(framerate, quality, param))


def variance_vs_param(df, param, reals, flags):
    var = []
    for acf in reals:
        var.append(np.var(acf))
    m = np.array(flags) > 0
    var = np.array(var)
    plt.clf()
    plt.figure(figsize=(20,15))
    plt.plot(df[param].values[m], np.log10(var[m]), ".", ms=10)
    if param == "age":
        plt.xlabel("$\mathrm{Age~[Gyr]}$")
    elif param == "teff":
        plt.xlabel = "$T_{\mathrm{eff}}~[K]$"
        plt.xlim(4500, 7000)
    elif param == "logg":
        plt.xlabel = "$\log(g)$"
    plt.ylabel("$\log_{10}(\mathrm{Variance})$")
    plt.savefig("{}_vs_variance".format(param))


if __name__ == "__main__":
    df = pd.read_csv("training_labels.csv")

    # param = "age"
    # param = "teff"
    param = "logg"
    Nstar = 525
    reals, flags = load_real_ifft(df, Nstar)
    make_animation(df, reals, flags, param, n_pixels=100, Nstar=Nstar,
                   linefit=False)
    save_as_movie(param)
    # variance_vs_param(df, param, reals, flags)
