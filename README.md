### Learn ImageProcess
Contains learning records of Tensorflow and Image processing in python
Hope this repository can be applied on the quadrotor and combine with flight control part.
0. build the python environment
1. consider about the CNN and transformer
2. consider the two-stream DNN for action and motion detection
3. consider the combination with the flight control of Drones

* purpose: To detect wildfire, smoke segementation

02-04-2020:
For smoke semantic segmentation, FCN, U-Net, and Seg-Net can be used.\\
Rough set and color character can be combined to use. **UKF** can be used to update the background.

03-29-2020:
Preparing to add the Q learning & Reinforcementlearning for Dornes.

03-31-2020:
Changed the learningrate picking method in `Smokesegmentation1.ipynb` with the help of fast.ai: https://course.fast.ai/videos/?lesson=5 46:30, method of Sylvian. The ACC increased from 85% to 89%.  
Next step is going to separate the fog and smoke.  

04-04-2020:
New package `Pythonrobotic` may be used. Pathplanning:grid algorithm or A star algorithm.
