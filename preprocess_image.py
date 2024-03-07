import matplotlib.pyplot as plt
from torchvision.io import read_image


image = read_image("data/PennFudanPed/PNGImages/FudanPed00016.png")
mask = read_image("data/PennFudanPed/PedMasks/FudanPed00016_mask.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
# Matplotlib 希望通道是图像张量的最后一个维度(而在 PyTorch 中它们是第一个维度) ，
# 因此我们将使用.permute方法把通道移动到图像的最后一个维度。
# 另注：opencv 里图像的存储为 BGR 格式，刚好和现在流行的 RGB 反过来了。以后可能要注意一下。
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
plt.show()