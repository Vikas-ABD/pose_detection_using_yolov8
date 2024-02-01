from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

model = YOLO(r'runs\runs\pose\train2\weights\best.pt')

results = model("test.jpg")

print(results)

r = results[0]
im_array = r.plot()  # plot a BGR numpy array of predictions
im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
# Display the image using matplotlib
plt.imshow(im)
plt.show()