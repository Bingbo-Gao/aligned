*******1*******
(1) The ./data/ folder contains the aligned heavy metal data  for the firse case study.
**************

******2*******
(2) The  ./originalGCCM/ folder contains codes of the GCCM published with the original paper with R language, but revised to run the new adligned data set. With ./GCCM/all/ contains  codes for isotropic case, and ./GCCM/nw/. contains  codes for anisotropic case.  ./originalGCCM/results/  contains corresponding results. 
*******

*******3*******
(3) The  ./periodicity/ folder contains  results of ./periodicityAll.py，which is used to  test the periodicity of all the simulated case of Katharina et al (the second case) 
**************

*******4*******
(4) The  ./rTrend/ folder contains results of the so called "improved version of  GCCM" by  Katharina et al,  with the linear trends removed for the aligned heavy metal data. The ./rTrend/differentK/ contains the results with different K parameter(number of repetitions of supsampling), the file names  end with K=5 are results by setting K=5 (the same as Katharina et al), and and file names  end with K=50 are results by setting to make the results more stable .
**************

*******5*******
(5) The ./alignedDataK.py contains codes of improved version of  GCCM of  Katharina et al,with the linear trends removed for the aligned heavy metal data. In this code we set K= 5 (the same as Katharina et al) and K=50 (to make the results more stable), and  two set of output locate under  ./rTrend/differentK/ 
**************


*******6*******
(6) The ./simulation.py contains codes to test the correlation of simulated data by Katharina et al. By calling the simulation codes of  Katharina et al, these codes generate 100 pairs of X and Y, and calculate their Pearson correlation and p-value.
**************

*******7*******
(7)./periodicityAll.py are code to test the periodicity of all the the simulated data of Katharina et al. The outputs locate under ./periodicity/ folder.    ./periodicityPart.py are codes to  in investigate the periodicity in detail. The resulst contains more information of the test results:       
	original_matrix
	detrended_matrix
	power_spectrum
	log_power_spectrum
	radial_power
	smoothed_radial
	peaks
	spatial_periods
	peak_frequencies
	angular_power
	angular_std
	angle_bins
	dominant_angle
	max_radius
Because the ouput is very large, we reduce all  data in ./periodicityAll.py , by revising the c_list from  [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.11, 0.12, 0.13, 0.14]  to [0, 0.04, 0.08,0.12], and revising the each repetition from 50 to 2. 
**************


*******8*******
(8)./GCCM_gao_corrected.py,./GCCM_sampling.py,./basic_gao.py, ./optimalEmbedding.py and ./optimalEmbedding_sampling.py are codes by Katharina et al to implement what they called impoved  GCCM. The  ./alignedDataK.py  called these codes.

./ diffusion.py and  ./diffusion.py are codes by Katharina et al to generate simulated data  for their second case.
**************
