data = read.csv('data/DailyDelhiClimateTrain.csv')

data$date = NULL
data$wind_speed = NULL
data$meantemp = NULL
summary(data$meanpressure)


data = ts(data, start=c(2013,01,01), frequency = 365)
plot(data)
par(mfrow=c(4,1))

plot(data[,1])
plot(data[,2])
plot(data[,3])
plot(data[,4], ylim = c(985,1030))
