setwd("E:\\Study\\R\\spatial-causality\\original\\raster")

# library(GSIF)
 library(rgdal)
 library(raster)
# library(gstat)
# library(ranger)
# library(scales)
# library(sp)
# library(lattice)
# library(gridExtra)
# library(intamap)
# library(maxlike)
# library(spatstat)
# library(entropy)
# library(gdistance)
# library(DescTools)

library(parallel)
library(foreach)
library(doParallel)

source("basic.r")
source("GCCM.r")


xImage<-readGDAL("cutdTRI.tif")     #read the casue variable 

xMatrix<-as.matrix(xImage)

imageSize<-dim(xMatrix)
totalRow<-imageSize[1]
totalCol<-imageSize[2]

x<- as.array(t(xMatrix))

print(x[locate(1,3,totalRow,totalCol)]) 

print(x[locate(2,5,totalRow,totalCol)]) 

print(x[locate(9,2,totalRow,totalCol)]) 


