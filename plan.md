# Image processing

- Convert all train images to JPEG/PNG
- Find max width & height for images
- Take max of width and height found and use as feature matrix size, max_size
- Convert all images to max_size
  - add padding if needed (add padding of value 0 -- black)
  - here we assume that a pixel of value 0 is uninteresting
- Get minimum and maximum width and heights for labeled bounding boxes

# ML Pipeline

- Split stage_1_train_images set into train_dev and train_test images with a 70:30 split
- Parse stage_1_train_labels.csv into:
  - matrix that maps to shape (size of train_dev, 5)
  - matrix that maps to shape (size of train_test, 5)
- Construct CNN

  - possibly model after AlexNet architecture w.r.t. pooling layers, filters, padding size, stride, etc

- Run train_dev images t
