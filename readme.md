# Optical Character Recognition Based on Sliding Convolution Attention Network[1]

+ Environment
  + Python 3.7 + pytorch 1.1.0  
+ Network
  + Currently implement based on Sliding Covolution Character Models[2]
  + Will transfer to SCAN then
+ Dataset
  + Train and Test on ICDAR
  + Might use IIIT5k for cross test
  + Augmentation
    + Rotation
    + Shear  
+ Loss
  + CTC Loss
    + [x] Pytorch  
    + [ ] Self-implement
+ Evaluation
+ Train
  + prepare environment
    + pytorch,opencv etc.
    + modify system settings in config
 
  ``` 
  python train.py 
  ```