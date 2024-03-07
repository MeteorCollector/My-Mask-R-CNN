import torch
import finetune
from mydataset import get_transform
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
# get the model using our helper function
model = finetune.get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load('mymask.pth'))
# 如果出现RuntimeError: input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
# 则注释掉下一行。
model.cuda()

image = read_image("data/PennFudanPed/PNGImages/FudanPed00016.png")
eval_transform = get_transform(train=False)

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]


image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.show()