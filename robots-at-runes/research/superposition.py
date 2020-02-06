import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

background = cv2.imread("images_sup/back1.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# logo = cv2.imread("images_sup/logo.png", cv2.IMREAD_UNCHANGED)
# logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)

logo = cv2.imread("images_sup/logo.png")
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)

h,w,_ = background.shape
logo = cv2.resize(logo, (w,h), interpolation = cv2.INTER_AREA)

composition = cv2.addWeighted(background,0.7,logo,0.3,0)

plt.imshow(background)
plt.show()
plt.imshow(logo)
plt.show()
plt.imshow(composition)
plt.show()