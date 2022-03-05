# Solving full-wave-nonlinear-inverse-scattering-problems with back-propagation-scheme
This Matlab code is used to solve inverse scattering problem with convolutional neural network by BPS.
Copyright © 2019, National University of Singapore, Zhun Wei. weizhun1010@gmail.com

More information, please welcome to visist: https://person.zju.edu.cn/0020060

Please first download matconvnet 1.0-beta23 : http://www.vlfeat.org/matconvnet/  remember to upzip it. To use mex, you also need visual studio installed.

(1) The Matlab code is used to implement back-propagation scheme (BPS) in
Z. Wei and X. Chen, “Deep learning schemes for full-wave nonlinear inverse scattering problems,” IEEE Transactions on Geoscience and Remote Sensing, 57 (4), pp. 1849-1860, 2019. 
 This Matlab code is used to solve inverse scattering problem with convolutional neural network by BPS, which is written by Zhun WEI (weizhun1010@gmail. com). 
Please feel free to contact if you have any question. Only CPU is required, and you can easily adapt it into GPU version or Python version.

(2) After training, you can simplely test the trained network by running “Display_Results_all_Results” for 25 examples. 
If you want to test it on a profile defined by yourself, you can define your profile in “data_generate_Circle_Es_S1” , run it, and then run “Data_generate_Circle_BP_S1” to generate BP inputs, and at last run “Display_Results_your_example”

(3) To start a new training: You can start your training by simple run “BPS_Training”, it consists of “data_generate_Circle_Es” (generate training scattering field) , “Data_generate_Circle_BP” (generate BP inputs), and training process.
To test your trained network, please refer to (2).

If you find this code is useful for you, please cite the following references:

Z. Wei and X. Chen, “Deep learning schemes for full-wave nonlinear inverse scattering problems,” IEEE Transactions on Geoscience and Remote Sensing, 57 (4), pp. 1849-1860, 2019. 

Z. Wei# and X. Chen, “Uncertainty Quantification in Inverse Scattering Problems with Bayesian Convolutional Neural Networks” IEEE Transactions on Antennas and Propagation, 10.1109/TAP.2020.3030974, 2020.

K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep convolutional neural network for inverse problems in imaging,” IEEE Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 2017.





