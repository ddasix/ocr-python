# %%
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# Reading the image
img = cv2.imread('./OCR_with_Python/Images/test01.jpg')

plt.imshow(img) # BGR

# %%
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
plt.imshow(rgb)

# %%
text = pytesseract.image_to_string(rgb)
print(text)

# %%



