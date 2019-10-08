import numpy as np

def _compute_SNR(x,y, message=''):
  noise = x - y
  noise_var = np.sum(noise ** 2)/len(noise) + 1e-7
  signal_energy = np.sum(y ** 2)/len(y)
  max_signal_energy = np.amax(y ** 2)
  SNR = 10 * np.log10(signal_energy/noise_var)
  PSNR = 10 * np.log10(max_signal_energy/noise_var)   
  print('{} SNR: {} PSNR: {}'.format(message, SNR, PSNR))
