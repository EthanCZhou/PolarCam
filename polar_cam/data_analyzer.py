import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mpl_toolkits.mplot3d.axes3d as axes3d

class DataAnalyzer:
    def __init__(self, nperseg=400, nfft=1600, windowsize=400, overlap=200):
        self.nperseg = nperseg
        self.nfft = nfft
        self.windowsize = windowsize
        self.overlap = overlap

    def analyze(self, intensities, timestamps, spot_id, output_directory):

        c90 = np.array(intensities['90'])
        c45 = np.array(intensities['45'])
        c135 = np.array(intensities['135'])
        c0 = np.array(intensities['0'])

        I0 = (c0 - c90) / (c0 + c90)
        I1 = (c45 - c135) / (c45 + c135)
        ANIS = I0 + 1j * I1
        ITOT = c90 + c0 + c45 + c135

        signals = {
            'I0': I0,
            'I1': I1,
            'ANIS': ANIS,
            'ITOT': ITOT
        }

        refractive_index = 1.333
        numerical_aperture = 1.2
        # alpha = np.arcsin(numerical_aperture/refractive_index)
        alpha = np.pi/2

        self.fft_welch(signals, timestamps, spot_id, output_directory)
        self.angle_plots(c0, c45, c90, c135, alpha, spot_id, output_directory)

    def fft_welch(
            self, signals, timestamps, spot_id, output_directory, threshold=1):
        plt.figure(figsize=(14, 8))

        for label, intensity in signals.items():
            n_seg = int((len(intensity) - self.overlap) / self.overlap)
            dom_freq = []
            current_time_segments = []

            for i in range(n_seg):
                start = i * self.overlap
                end = start + self.windowsize
                segment = intensity[start:end]

                if len(segment) < self.windowsize:
                    continue

                segment_timestamps = timestamps[start:end]
                fs = 1 / np.mean(np.diff(segment_timestamps))

                is_complex = np.iscomplexobj(segment)
                freqs, power = signal.welch(
                    segment, 
                    fs=fs, 
                    nperseg=self.nperseg, 
                    nfft=self.nfft, 
                    return_onesided=not is_complex
                )
                magnitude = np.abs(power)

                if np.max(magnitude) >= threshold:
                    dominant_frequency = freqs[np.argmax(magnitude)]
                    dom_freq.append(dominant_frequency)
                    current_time_segments.append(
                        (segment_timestamps[0] + segment_timestamps[-1]) / 2
                    )

            if dom_freq:
                plt.plot(current_time_segments, dom_freq, label=label)

        plt.title(f'Speed-Time Diagram for Spot {spot_id}')
        plt.xlabel('Time (s)')
        plt.ylabel('Dominant Frequency (Hz)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(
            output_directory, f'speed_time_diagram_spot_{spot_id}.png')
        plt.savefig(plot_filename)
        plt.close()

    def angle_plots(self, c0, c45, c90, c135, alpha, spot_id, output_directory):
        A = (1/6) - (np.cos(alpha)/4) + (((np.cos(alpha))**3)/12)
        B = (np.cos(alpha)/8) - (((np.cos(alpha))**3)/8)
        C = (7/48) - (np.cos(alpha)/16) - (((np.cos(alpha))**2)/16) - (((np.cos(alpha))**3)/48)

        phis = []
        thetas = []
        ixs = []
        iys = []
        intensities = []

        x = []
        y = []
        z = []

        for i in range(len(c90)):

            phi  = 0.5 * np.arctan2((c45[i] / 2 - c135[i] / 2), (c0[i] / 2 - c90[i] / 2))
            
            phis.append(phi)

            cs = np.cos(2 * phi)
            ss = np.sin(2 * phi)

            OP = c0[i] + c45[i] + c90[i] + c135[i]
            P = c0[i] - c90[i] + c45[i] - c135[i]

            intensities.append(OP)

            sinsqtheta = 4 * A * P / (2 * (ss + cs) * OP * C - 4 * B * P)
            test = np.sqrt(np.abs(sinsqtheta))
            test = np.clip(test, 0, 1)
            theta = np.arcsin(test)

            thetas.append(theta)

            x.append(np.sin(phi) * np.cos(theta))
            y.append(np.sin(phi) * np.sin(theta))
            z.append(np.cos(theta))

            ix = (c0[i] - c90[i]) / (c0[i] + c90[i])
            iy = (c45[i] - c135[i]) / (c45[i] + c135[i])

            ixs.append(ix)
            iys.append(iy)

        phi_unwrapped = np.unwrap(2 * phis)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(2,1,1, projection='3d')
        ax2 = fig1.add_subplot(2,1,2)
        plot1 = ax1.scatter3D(np.array(x), np.array(y), np.array(z))
        plot2 = ax2.scatter(np.array(x), np.array(y))
        plot_filename_1 = os.path.join(
            output_directory, f'polar_plot_{spot_id}.png')
        plt.savefig(plot_filename_1)
        plt.close()

        fig2 = plt.figure(figsize=(12, 4))
        plt.plot(phis, lw=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Phi (rad)")
        plt.title("Fourkas Azimuthal Angle φ Over Time")
        plt.grid(True)
        plot_filename_2 = os.path.join(
            output_directory, f'angle_over_time_{spot_id}.png')
        plt.savefig(plot_filename_2)
        plt.close()

        fig3 = plt.figure(figsize=(6, 6))
        plt.scatter(ixs, iys, s=1, alpha=0.5)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Anisotropy Orbit Scatter")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid(True)
        plt.tight_layout()
        plot_filename_3 = os.path.join(
            output_directory, f'anisotropy_{spot_id}.png')
        plt.savefig(plot_filename_3)
        plt.close()

        fig4 = plt.figure(figsize=(6, 6))
        ax1 = fig4.add_subplot(1,1,1, projection='3d')
        colors = np.array(intensities)
        plot4 = ax1.scatter3D(np.array(ixs), np.array(iys), np.array(intensities), c=colors, cmap='viridis')
        plot_filename_4 = os.path.join(
            output_directory, f'anisotropy3D_{spot_id}.png')
        plt.savefig(plot_filename_4)
        plt.close()
        