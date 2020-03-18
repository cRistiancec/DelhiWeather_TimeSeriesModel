data = read.csv('data/DailyDelhiClimateTrain.csv')



# data preparation --------------------------------------------------------



data$date = NULL

summary(data)

# time series creation  --------------------------------------------------------

data = ts(data, start=c(2013,01,01), frequency = 365)
plot(data, main = 'Daily weather in New Delhi')


# graphical exploration  --------------------------------------------------------

par(mfrow=c(4,1))

#plot(data[,1])
#plot(data[,2])
#plot(data[,3])
#plot(data[,4],ylim=c(980,1030))

temp_decomp = decompose(data[,1])
humidity_decomp = decompose(data[,2])
wind_decomp = decompose(data[,3])
pressure_decomp = decompose(data[,4])

plot(temp_decomp)
plot(humidity_decomp)
plot(wind_decomp)
plot(pressure_decomp)

#using Acf and Pacf (forecast library) instead of acf and pacf gives better results

acf(data[,1],lag.max=1000, main = 'Mean temperature')
acf(data[,2], lag.max=1000, main = 'Mean humidity')
acf(data[,3], lag.max=1000, main = 'Wind speed')
acf(data[,4], lag.max=1000, main = 'Mean pressure')

par(mfrow=c(4,1))

pacf(data[,1], main = 'Mean temperature')
pacf(data[,2], main = 'Mean humidity')
pacf(data[,3], main = 'Wind speed')
pacf(data[,4], main = 'Mean pressure')

# 9. Ljung-Bx test
Box.test(data[,1])
 
Box.test(data[,2])

Box.test(data[,3])

Box.test(data[,4])

#Now seasonalities



plot(diff(data[,1],365), main ='Seasonal mean temperature')
abline(h= 0, col = 'red')
abline(h= 2.5, col = 'blue')
#Last part of 2016 seems hotter on average than the previous year, also it explainds the trend.

plot(diff(data[,2],365),main ='Seasonal mean humidity')
abline(h= 0, col = 'red')
plot(diff(data[,3],365),main ='Seasonal mean wind speed')
abline(h= 0, col = 'red')
plot(diff(data[,4],365),main ='Seasonal mean pressure')
abline(h= 0, col = 'red')

par(mfrow=c(4,1))

acf(diff(data[,1],365),main ='Seasonal mean temperature')
acf(diff(data[,2],365),main ='Seasonal mean humidity')
acf(diff(data[,3],365),main ='Seasonal mean wind speed')
acf(diff(data[,4],365),main ='Seasonal mean pressure')

pacf(diff(data[,1],365),main ='Seasonal mean temperature')
pacf(diff(data[,2],365),main ='Seasonal mean humidity')
pacf(diff(data[,3],365),main ='Seasonal mean wind speed')
pacf(diff(data[,4],365),main ='Seasonal mean pressure')


# Testing area ------------------------------------------------------------



var(data[,1])

Acf(data[,1])
Pacf(data)

tso(log(data[,4]),types=c("AO"))


plot(density(diff(data[,1])))
plot(diff(diff(data[,1])))

library(tsoutliers)
library(tidyverse)
library(forecast)
library(factoextra)


cl <- parallel::makeCluster(8)
doParallel::registerDoParallel(cl)

fit<-auto.arima(data[,1],max.order = 10)

parallel::stopCluster(cl)

data[,1] %>%
  Arima(order=c(3,1,1), seasonal=c(0,1,0)) %>%
  residuals() %>% ggtsdisplay()


#ARIMA(3,1,1)(0,1,0)[365]

# Outlier detection ------------------------------------------------------------

##Outlier detection with kmeans (adds more flavour)
require(Ckmeans.1d.dp)

#Temp

res <- Ckmeans.1d.dp(data[,1], k=c(1:10))
plot(res)


plot(data[,1], main = "Time course clustering / peak calling", 
     col=res$cluster, pch=res$cluster, type="h", 
     xlab="Time t", ylab="Transformed intensity w")
abline(v=res$centers, col="chocolate", lty="dashed")
text(res$centers, max(10) * .95, cex=0.75, font=2,
     paste(round(res$size / sum(res$size) * 100), "/ 100"))


ckm <- Ckmeans.1d.dp(data[,1], k=length(res$size))
midpoints <- ahist(ckm, style="midpoints", data=data[,1], plot=FALSE)$breaks[2:length(res$size)]

plot(ckm, main="Midpoints as cluster boundaries")
abline(v=midpoints, col="RoyalBlue", lwd=3)
legend("topright", "Midpoints", lwd=3, col="RoyalBlue")


cl <- parallel::makeCluster(9)
doParallel::registerDoParallel(cl)

fit<-auto.arima(data[,1],max.order = 10)

parallel::stopCluster(cl)

fit_saved = fit
# Series: data[, 1] 
# Regression with ARIMA(1,1,2)(0,1,0)[365] errors 
# 
# Coefficients:
#   ar1      ma1      ma2    AO79    AO162   TC709   AO905   AO1246
# 0.6991  -0.9037  -0.0852  6.5261  -7.8627  5.8848  8.2687  -8.6357
# s.e.  0.0318   0.0413   0.0391  1.5774   1.5765  1.4172  1.1178   1.5761
# 
# sigma^2 estimated as 4.096:  log likelihood=-2328.35
# AIC=4674.71   AICc=4674.87   BIC=4719.7
# 
# Outliers:
#   type  ind     time coefhat  tstat
# 1   AO   79  2013:79   6.526  4.137
# 2   AO  162 2013:162  -7.863 -4.987
# 3   TC  709 2014:344   5.885  4.152
# 4   AO  905 2015:175   8.269  7.398
# 5   AO 1246 2016:151  -8.636 -5.479

# define the variables containing the outliers for
# the observations outside the sample
npred <- 24 # number of periods ahead to forecast 
newxreg <- outliers.effects(fit$outliers, length(data[,1]) + npred)
newxreg <- ts(newxreg[-seq_along(data[,1]),], start = c(2013, 1))

# obtain the forecasts
p <- predict(fit$fit, n.ahead=npred, newxreg=newxreg)


plot(cbind(data[,1], p$pred), plot.type = "single", ylab = "", type = "n")
lines(data[,1])
lines(p$pred, type = "l", col = "blue")
lines(p$pred + 1.96 * p$se, type = "l", col = "red", lty = 2)  
lines(p$pred - 1.96 * p$se, type = "l", col = "red", lty = 2)  
legend("topleft", legend = c("observed data", 
                             "forecasts", "95% confidence bands"), lty = c(1,1,2,2), 
       col = c("black", "blue", "red", "red"), bty = "n")


#Again with no outliers

cl <- parallel::makeCluster(9)
doParallel::registerDoParallel(cl)

fit_dos<- auto.arima(data[,1],max.order = 10)

parallel::stopCluster(cl)

npred <- 24 # number of periods ahead to forecast 

# obtain the forecasts
p <- predict(fit_dos, n.ahead=npred)


plot(cbind(data[,1], p$pred), plot.type = "single", ylab = "", type = "n")
lines(data[,1])
lines(p$pred, type = "l", col = "blue")
lines(p$pred + 1.96 * p$se, type = "l", col = "red", lty = 2)  
lines(p$pred - 1.96 * p$se, type = "l", col = "red", lty = 2)  
legend("topleft", legend = c("observed data", 
                             "forecasts", "95% confidence bands"), lty = c(1,1,2,2), 
       col = c("black", "blue", "red", "red"), bty = "n")
#Results are the same almost so, only those that have clear outliers in the cluster analysis will be used.


#Humidity

res <- Ckmeans.1d.dp(data[,2], k=c(1:10))
plot(res)


plot(data[,2], main = "Time course clustering / peak calling", 
     col=res$cluster, pch=res$cluster, type="h", 
     xlab="Time t", ylab="Transformed intensity w")
abline(v=res$centers, col="chocolate", lty="dashed")
text(res$centers, max(1) * .95, cex=0.75, font=2,
     paste(round(res$size / sum(res$size) * 100), "/ 100"))


ckm <- Ckmeans.1d.dp(data[,2], k=length(res$size))
midpoints <- ahist(ckm, style="midpoints", data=data[,2], plot=FALSE)$breaks[2:length(res$size)]

plot(ckm, main="Midpoints as cluster boundaries")
abline(v=midpoints, col="RoyalBlue", lwd=3)
legend("topright", "Midpoints", lwd=3, col="RoyalBlue")

cl <- parallel::makeCluster(9)
doParallel::registerDoParallel(cl)

fit_dos<- auto.arima(data[,2],max.order = 10)

parallel::stopCluster(cl)

npred <- 24 # number of periods ahead to forecast 

# obtain the forecasts
p <- predict(fit_dos, n.ahead=npred)


plot(cbind(data[,2], p$pred), plot.type = "single", ylab = "", type = "n")
lines(data[,2])
lines(p$pred, type = "l", col = "blue")
lines(p$pred + 1.96 * p$se, type = "l", col = "red", lty = 2)  
lines(p$pred - 1.96 * p$se, type = "l", col = "red", lty = 2)  
legend("topleft", legend = c("observed data", 
                             "forecasts", "95% confidence bands"), lty = c(1,1,2,2), 
       col = c("black", "blue", "red", "red"), bty = "n")



#Wind Speed

res <- Ckmeans.1d.dp(data[,3], k=c(1:10))
plot(res)


plot(data[,3], main = "Time course clustering / peak calling", 
     col=res$cluster, pch=res$cluster, type="h", 
     xlab="Time t", ylab="Transformed intensity w")
abline(v=res$centers, col="chocolate", lty="dashed")
text(res$centers, max(1) * .95, cex=0.75, font=2,
     paste(round(res$size / sum(res$size) * 100), "/ 100"))


ckm <- Ckmeans.1d.dp(data[,3], k=length(res$size))
midpoints <- ahist(ckm, style="midpoints", data=data[,3], plot=FALSE)$breaks[2:length(res$size)]

plot(ckm, main="Midpoints as cluster boundaries")
abline(v=midpoints, col="RoyalBlue", lwd=3)
legend("topright", "Midpoints", lwd=3, col="RoyalBlue")

cl <- parallel::makeCluster(9)
doParallel::registerDoParallel(cl)

fit_tres<- auto.arima(data[,3],max.order = 10)

parallel::stopCluster(cl)

npred <- 24 # number of periods ahead to forecast 

# obtain the forecasts
p <- predict(fit_tres, n.ahead=npred)


plot(cbind(data[,3], p$pred), plot.type = "single", ylab = "", type = "n")
lines(data[,3])
lines(p$pred, type = "l", col = "blue")
lines(p$pred + 1.96 * p$se, type = "l", col = "red", lty = 2)  
lines(p$pred - 1.96 * p$se, type = "l", col = "red", lty = 2)  
legend("topleft", legend = c("observed data", 
                             "forecasts", "95% confidence bands"), lty = c(1,1,2,2), 
       col = c("black", "blue", "red", "red"), bty = "n")


#Pressure

res <- Ckmeans.1d.dp(data[,4], k=c(1:10))
plot(res)


plot(data[,4], main = "Time course clustering / peak calling", 
     col=res$cluster, pch=res$cluster, type="h", 
     xlab="Time t", ylab="Transformed intensity w")
abline(v=res$centers, col="chocolate", lty="dashed")
text(res$centers, max(1) * .95, cex=0.75, font=2,
     paste(round(res$size / sum(res$size) * 100), "/ 100"))


ckm <- Ckmeans.1d.dp(data[,4], k=length(res$size))
midpoints <- ahist(ckm, style="midpoints", data=data[,4], plot=FALSE)$breaks[2:length(res$size)]

plot(ckm, main="Midpoints as cluster boundaries")
abline(v=midpoints, col="RoyalBlue", lwd=2)
abline(v=1013, col = 'red',lwd = 3)
legend('topright', c('Midpoints', "acceptable values"), lwd = 3, col = c('RoyalBlue','red'))

#Clearly there are outliers

cl <- parallel::makeCluster(9)
doParallel::registerDoParallel(cl)

fit_tres<- tso(data[,3],types=c("AO,IO,TS,LS"))
fit_cuatro<- tso(log(data[,4]),types=c("AO"))

parallel::stopCluster(cl)


# define the variables containing the outliers for
# the observations outside the sample
npred <- 24 # number of periods ahead to forecast 
newxreg <- outliers.effects(fit_cuatro$outliers, length(data[,4]) + npred)
newxreg <- ts(newxreg[-seq_along(data[,4]),], start = c(2013, 1))

# obtain the forecasts
p <- predict(fit$fit, n.ahead=npred, newxreg=newxreg)


plot(cbind(data[,4], p$pred), plot.type = "single", ylab = "", type = "n")
lines(data[,4])
lines(p$pred, type = "l", col = "blue")
lines(p$pred + 1.96 * p$se, type = "l", col = "red", lty = 2)  
lines(p$pred - 1.96 * p$se, type = "l", col = "red", lty = 2)  
legend("topleft", legend = c("observed data", 
                             "forecasts", "95% confidence bands"), lty = c(1,1,2,2), 
       col = c("black", "blue", "red", "red"), bty = "n")
