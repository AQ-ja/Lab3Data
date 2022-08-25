# Importacion de las librerias 
library("readxl")
library(keras)
library(tensorflow)
library(dplyr)
library(forecast)
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

#Se normaliza para que el algoritmo trabaje mejor
norm_s1<-scale(s1)


#Para que pueda usarse el argoritmo LSTM es necesario transformar la serie en una supervisada.
lgd <-c(rep(NA, 1),norm_s1[1:(length(norm_s1)-1)])
supervised <-as.data.frame(cbind(lgd,norm_s1))
colnames(supervised)<-c("x-1", "x")
supervised[is.na(supervised)]<-0



#Dividir la serie 60% para entrenamiento, 20% para validación y 20% para prueba
entrenamiento<-round(0.6*length(s1))
val_test<-round(0.2*length(s1))
test<-tail(supervised, val_test)
supervised<-supervised %>% head(nrow(.)-val_test)
validation<-supervised %>% tail(val_test)
supervised<-head(supervised, nrow(supervised)-val_test)
#El train son los que quedan
train<-supervised
rm(supervised)


# Division de entrenamiento y validacion 
# Entrenamiento
y_train <- train[,2]
x_train <- train[,1]
y_val <- validation[,2]
x_val <- validation[,1]
y_test <- test[,2]
x_test <- test[,1]

#Convertir a matricez de 3 dimenciones 
paso <- 1
caracteristicas<-1 #es univariada
dim(x_train) <- c(length(x_train),paso,caracteristicas)
dim(y_train) <- c(length(y_train),caracteristicas)
dim(x_test) <- c(length(x_test),paso,caracteristicas)
dim(y_test) <- c(length(y_test),caracteristicas)
dim(x_val) <- c(length(x_val),paso,caracteristicas)
dim(y_val) <- c(length(y_val),caracteristicas)



#CONSTRUCION DEL MODELO 1
lote = 1
unidades <- 1
modelo1 <-keras_model_sequential()
modelo1 %>% 
  layer_lstm(unidades, batch_input_shape=c(lote, paso, caracteristicas),
             stateful = T) %>% 
  layer_dense(units = 1)

summary(modelo1)

#-------------------------------------------------------------------------------

modelo1 %>%
  compile(
    optimizer = 'rmsprop',
    loss = 'mse'
  )

#-------------------------------------------------------------------------------
# Entrenar el modelo 
epocas <- 50
history <- modelo1 %>% fit(
  x = x_train,
  y = y_train,
  validation_data = list(x_val, y_val),
  batch_size = lote,
  epochs = epocas,
  shuffle = FALSE,
  verbose = 0
)

plot(history)

#-------------------------------------------------------------------------------
#Evaluar el modelo 
print("Entrenamiento")
modelo1 %>% evaluate(
  x = x_train,
  y = y_train
)
print("Validación")
modelo1 %>% evaluate(
  x = x_val,
  y = y_val
)
print("Prueba")
modelo1 %>% evaluate(
  x = x_test,
  y = y_test
)

#-------------------------------------------------------------------------------
#PREDICCION DEL MODELO 1
prediccion_fun <- function(data,modelo, batch_size,scale,center,dif=F, Series=NULL,n=1){
  prediccion <- numeric(length(data))
  if (dif==F){
    for(i in 1:length(data)){
      X = data[i]
      dim(X) = c(1,1,1)
      yhat = modelo %>% predict(X, batch_size=batch_size)
      # invert scaling
      yhat = yhat*scale+center
      # store
      prediccion[i] <- yhat
    }
  }else{
    for(i in 1:length(data)){
      X = data[i]
      dim(X) = c(1,1,1)
      yhat = modelo1 %>% predict(X, batch_size=batch_size)
      # invert scaling
      yhat = yhat*scale+center
      # invert differencing
      yhat  = yhat + Series[(n+i)]
      # store
      prediccion[i] <- yhat
    }
  }
  
  return(prediccion)
}

prediccion_val <- prediccion_fun(x_val,modelo1,1,attr(norm_s1,"scaled:scale"),
                                 attr(norm_s1,"scaled:center"),dif=T,s1,entrenamiento              
)
prediccion_test <- prediccion_fun(x_test,modelo1,1,attr(norm_s1,"scaled:scale"),
                                  attr(norm_s1,"scaled:center"),dif=T,s1,entrenamiento+val_test               
)

#--------------------------------------------------------------------------------------------------
# Graficando el modelo 1
serie<-s1
serie_test<- tail(serie,val_test)
serie<-head(serie,length(serie)-val_test)
serie_val<-tail(serie,val_test)
serie<-head(serie,length(serie)-val_test)
serie_train <- serie


df_serie_total<-data.frame(pass=as.matrix(s1), date=zoo::as.Date(time(s1)))
df_serie_val<-data.frame(pass=prediccion_val, date=zoo::as.Date(time(serie_val)))
df_serie_test<-data.frame(pass=prediccion_test, date=zoo::as.Date(time(serie_test)))


df_serie_total$class <- 'real'
df_serie_val$class <- 'validacion'
df_serie_test$class <- 'prueba'

df_serie<-rbind(df_serie_total, df_serie_val,df_serie_test)
df_serie$class<-factor(df_serie$class,levels = c('real','validacion','prueba'))
ggplot(df_serie,aes(x = date, y = pass, colour = class)) +
  geom_line()




#/////////////////////////////////////////
#               MODELO 2
# ///////////////////////////////////////




