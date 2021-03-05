import numpy as np
import matplotlib.pyplot as plt

sub0 = np.load(f'0017_sub0_low.npy')
sub45 = np.load(f'0017_sub45_low.npy')

plt.close('all')
plt.figure(1)
plt.imshow(sub0)
plt.title('Shear0 Subtraction Image')

plt.figure(2)
plt.imshow(sub45)
plt.title('Shear45 Subtraction Image')