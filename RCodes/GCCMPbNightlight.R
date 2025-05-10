setwd("/work/home/ac1opnizd4")

library(parallel)
library(foreach)
library(doParallel)
source("basic.r")
source("GCCM.r")

load("CuPbCdMgEnvi.RData")

lib_sizes<-seq(10,120,20)
lib<-NULL

startTime<-Sys.time()

HMs<-c("Cu","Pb","Cd","Mg") 
Envs<-c("dTRI","nlights03")



E=1

yName<-HMs[2]
yVariable<-HMImages[[2]]
yMatrix<-as.matrix(yVariable)
imageSize<-dim(yMatrix)
totalRow<-imageSize[1]
totalCol<-imageSize[2]

y<-as.vector(yMatrix)


predRows<-seq(5,totalRow,5)
predCols<-seq(5,totalCol,5)
pred<-merge(predRows,predCols)


xName<-Envs[2]
xVariable<-EnviImages[[2]]
xMatrix<-as.matrix(xVariable)

x<-as.vector(xMatrix)

lmModel<-lm(y ~ x, data.frame(y,x))
prediction<-predict(lmModel,data.frame(x))
y<-y-prediction
yMatrixM<-matrix(y,nrow = totalRow, ncol = totalCol)



x_xmap_y <- GCCM(xMatrix, yMatrixM, lib_sizes, lib, pred, E,tau = 1,b=E+2,winStepRatio = 0,cores=30,dir=0,validRatio=0)
y_xmap_x <- GCCM(yMatrixM, xMatrix, lib_sizes, lib, pred, E,tau = 1,b=E+2,winStepRatio = 0,cores=30,dir=0,validRatio=0)

x_xmap_y$L <- as.factor(x_xmap_y$L)
x_xmap_y_means <- do.call(rbind, lapply(split(x_xmap_y, x_xmap_y$L), function(x){max(0, mean(x$rho,na.rm=TRUE))}))

y_xmap_x$L <- as.factor(y_xmap_x$L)
y_xmap_x_means <- do.call(rbind, lapply(split(y_xmap_x, y_xmap_x$L), function(x){max(0, mean(x$rho,na.rm=TRUE))}))

x_xmap_y_Sig<- significance(x_xmap_y_means,nrow(pred))    #Test the significance of the prediciton accuray
y_xmap_x_Sig<- significance(y_xmap_x_means,nrow(pred))     #Test the significance of the prediciton accuray

x_xmap_y_interval<- confidence(x_xmap_y_means,nrow(pred))
colnames(x_xmap_y_interval)<-c("x_xmap_y_upper","x_xmap_y_lower")   #calculate the  95%. confidence interval  of the prediciton accuray

y_xmap_x_interval<- confidence(y_xmap_x_means,nrow(pred))
colnames(y_xmap_x_interval)<-c("y_xmap_x_upper","y_xmap_x_lower")  #calculate the  95%. confidence interval  of the prediciton accuray

results<-data.frame(lib_sizes,x_xmap_y_means,y_xmap_x_means,x_xmap_y_Sig,y_xmap_x_Sig,x_xmap_y_interval,y_xmap_x_interval)  #Save the cross-mapping prediciton results
write.csv(results, file=paste(xName,"_",yName,"_Eis",E,".csv",sep = ""))
par(mfrow=c(1,1))
par(mar=c(5, 4, 4, 2) + 0.1)

jpeg(filename = paste(xName,"_",yName,"_Eis",E,".jpg",sep = ""),width = 600, height = 400)

plot(lib_sizes, x_xmap_y_means, type = "l", col = "royalblue", lwd = 2,
     xlim = c(min(lib_sizes), max(lib_sizes)), ylim = c(0.0, 1), xlab = "L", ylab = expression(rho))
lines(lib_sizes, y_xmap_x_means, col = "red3", lwd = 2)
# legend(min(lib_sizes), 1, legend = c("x xmap y", "y xmap x"),
#        xjust = 0, yjust = 1, lty = 1, lwd = 2, col = c("royalblue", "red3"))
legend(min(lib_sizes), 1, legend = c(paste(xName,"xmap",yName,sep=" "), paste(yName,"xmap",xName,sep=" ")), 
       xjust = 0, yjust = 1, lty = 1, lwd = 2, col = c("royalblue", "red3"))
dev.off()

endTime<-Sys.time()

print(difftime(endTime,startTime, units ="mins"))






