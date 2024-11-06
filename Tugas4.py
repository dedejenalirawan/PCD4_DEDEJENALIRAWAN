import imageio.v2 as imageio  # Menggunakan imageio.v2 untuk menghindari DeprecationWarning
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Memuat citra dalam mode grayscale
image = imageio.imread(r'E:\CODINGAN\PCD4\sasuke uchiha.jpg', mode='L')

# Fungsi untuk melakukan ekualisasi histogram
def histogram_equalization(img):
    # Hitung histogram dan CDF
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Normalisasi CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Aplikasikan ke citra
    img_equalized = cdf[img.astype('uint8')]
    return img_equalized

# Terapkan ekualisasi histogram
equalized_image = histogram_equalization(image)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Citra Awal')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Citra Hasil Ekualisasi Histogram')
plt.axis('off')
plt.show()
