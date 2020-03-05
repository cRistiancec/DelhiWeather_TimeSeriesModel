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

cl <- parallel::makeCluster(8)
doParallel::registerDoParallel(cl)

fit<-auto.arima(data[,1],max.order = 10)

parallel::stopCluster(cl)

data[,1] %>%
  Arima(order=c(3,1,1), seasonal=c(0,1,0)) %>%
  residuals() %>% ggtsdisplay()


#ARIMA(3,1,1)(0,1,0)[365]