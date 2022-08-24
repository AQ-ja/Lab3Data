# Importacion de las librerias 
library("readxl")
library(dplyr)
library(forecast)
library(keras)
library(tensorflow)
library(ggplot2)
library(recipes)
library(lubridate)

# Traer el archivo del lab pasado 
impor<-read_excel("Importacion.xlsx")

View(impor)

#Aplicacion de LSTM a la importacion del Diesel 
fd <- impor[c('Fecha','Diesel')]
diesel_ts <- ts(fd$Diesel, start = c(2001, 1),frequency = 12)
s1<-diff(diesel_ts)
plot(s1)

norm_s1<-scale(s1)

#Serie supervisada
lgd <-c(rep(NA, 1),norm_s1[1:(length(serie_norm)-1)])
supervised <-as.data.frame(cbind(lgd,norm_s1))
coln(supervised)<-c("x-1", "x")
supervised[is.na(supervised)]<-0

train<-round(0.6*length(s1))
val_test<-round(0.2*length(s1))

test<-tail(supervised, val_test)

#Cortamos la mattriz
supervised<-supervised %>% head(nrow9(.)-val_test)

