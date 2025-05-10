setwd("E:\\Study\\spatial_causality")

library("pcalg")

library(rgdal)
library(raster)


HMs<-c("Cu","Pb","Cd","Mg")

Envs<-c("dTRI","nlights03")

for(h in seq(1:length(HMs)))
{
  yImage<-readGDAL(paste("../data/",HMs[h],".tif", sep=""))
  yMatrix<-as.matrix(yImage)
  y<-as.vector(yMatrix)
  
  for(e in seq(1:length(Envs)))
  {
    xImage<-readGDAL(paste("../data/",Envs[e],".tif", sep=""))
    
    xMatrix<-as.matrix(xImage)
    x<-as.vector(xMatrix)
    
    X <- cbind(x,y)
    res <- lingam(X)
    
    print(paste("y=",HMs[h],",x=",Envs[e],sep=""))
    
    print(res)
    
    print(as(res, "amat")) 
    
  }
}