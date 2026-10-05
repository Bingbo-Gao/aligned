***1***
The ./data/ folder contain the heavy metal data aligned for the firse case study.
*******

***2***
The  ./originalGCCM/ folder contains codes of the GCCM published with the original paper with R language, but revised to run the new adligned data set. With ./GCCM/all/ contains  codes for isotropic case, and ./GCCM/nw/. contains  codes for anisotropic case.
*******

***3***
The  ./periodicity/ folder contains codes with python languge to test the periodicity in the simulated case of Katharina et al (the second case) 
*******


***4***
The  ./rTrend/folder contains results of improved version of  GCCM of  Katharina et al,  with the linear trends removed for the aligned heavy metal data. The ./rTrend/differentK/ contains the results with different K parameter(number of repetitions of supsampling), the file names  end with K=5 are results by setting K=5 (the same as Katharina et al), and and file names  end with K=50 are results by setting to make the results more stable .
*******

***4***
The ./alignedDataK=5.py contains codes of improved version of  GCCM of  Katharina et al,with the linear trends removed for the aligned heavy metal data. The alignedData.py use the default

