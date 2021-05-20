# Instructions for the calculation of the mAP, F1 Score and TP-FP-FN

First, clone the AlexeAB Darknet repo, then:
* Replace detector.c file with the new one then put this file into darknet/src directory
* Put _yolov4-obj.cfg_ file into the darknet/cfg directory
* Create a folder named _Test_ into darknet/data directory
* Place test images and labeled into the test folder
* Put generate_test.py file into darknet directory
  
## Notes
#### Minor changes at detector.c file
```bash
# in line #1103 for loop is modified for our 8 classes.
for (class_id = 0; class_id < 8; class_id++) 

# in line #1278 for loop is modified for calculation of the correct precision on 8 classes.
for (i = 0; i < 8; ++i) { double avg_precision = 0;
```


