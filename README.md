# Glass-and-Concrete-Crack-Recognition
Python code including the LBP feature extractor and SVM classification for crack and none-cracked glass and concrete images. 

Unified Local Binary Pattern ULBP features are extracted from grayscale images of glass pannels or concrete surfaces using Python code "LBP(58)_9SectorsIntensityHistogram_V1".
The extracted ULBP code is then used to train an SVM classifier for the use of cracked and none-cracked image predictions using python code "Svm4Cat_5Fold_LBP58_V1".

![image](https://github.com/faxirabd/Glass-and-Concrete-Crack-Recognition/assets/115953037/5d55a3df-d750-44de-9fbf-0c31311eec9c)

Partition based ULBP method:

This method calculates the Uniform Local Binary Pattern (ULBP) code for each of the input images. The input image is partitioned into 9 equal blocks and for each of the blocks, a histogram of ULBP is formed. The 9 histograms were normalized and concatenated before feeding into a classifier.

Glass and Concrete Crack Recognition Prototype:

![image](https://github.com/faxirabd/Glass-and-Concrete-Crack-Recognition/assets/115953037/16ef9ee8-eb0d-48c2-a6ed-656e7787b01b)

![image](https://github.com/faxirabd/Glass-and-Concrete-Crack-Recognition/assets/115953037/89bb6ac1-1006-4c18-91c5-26c1b937011e)
