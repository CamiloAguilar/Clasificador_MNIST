#****************************************************************************************************************
#********************************************  PArcial Final   **************************************************
#****************************************************************************************************************
options(scipen = 999)
library(dplyr); library(nnet)

## Función para graficar un número
show_digit <- function(numero, col = gray(12:1 / 12), ...) {
     image(matrix(as.matrix(numero[-785]), nrow = 28)[, 28:1], col = col, ...)
}

##*********************************
## 1. Lectura información
##*********************************

## Información de entrenamiento
## PAra un ejercicio pasado, se había ya estructurado la información.
## Un detalle completo de la obtenión de la información MNIST bajo esta estructura se puede
## revisar en https://github.com/CamiloAguilar/MNIST

## La siguientes líneas cargan la información de entrenamiento
Train_0 <- readRDS("./MNIST/Train_0.rds")
Train_1 <- readRDS("./MNIST/Train_1.rds")
Train_2 <- readRDS("./MNIST/Train_2.rds")
Train_3 <- readRDS("./MNIST/Train_3.rds")
Train_4 <- readRDS("./MNIST/Train_4.rds")
Train_5 <- readRDS("./MNIST/Train_5.rds")
Train_6 <- readRDS("./MNIST/Train_6.rds")
Train_7 <- readRDS("./MNIST/Train_7.rds")
Train_8 <- readRDS("./MNIST/Train_8.rds")
Train_9 <- readRDS("./MNIST/Train_9.rds")

## La siguientes líneas cargan la información de prueba
Test_0 <- readRDS("./MNIST/Test_0.rds")
Test_1 <- readRDS("./MNIST/Test_1.rds")
Test_2 <- readRDS("./MNIST/Test_2.rds")
Test_3 <- readRDS("./MNIST/Test_3.rds")
Test_4 <- readRDS("./MNIST/Test_4.rds")
Test_5 <- readRDS("./MNIST/Test_5.rds")
Test_6 <- readRDS("./MNIST/Test_6.rds")
Test_7 <- readRDS("./MNIST/Test_7.rds")
Test_8 <- readRDS("./MNIST/Test_8.rds")
Test_9 <- readRDS("./MNIST/Test_9.rds")

## A continuación unimos las bases, una para entrenamiento (train) y otra para pruebas (test) 
## Generamos además un vector con las etiquetas para cada imagen
train <- rbind(Train_0, Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9)

y_train <- train$label; train <- train[, 1:784]
rm(Train_0, Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9)

test <- rbind(Test_0, Test_1, Test_2, Test_3, Test_4, Test_5, Test_6, Test_7, Test_8, Test_9)
y_test <- test$label; test <- test[, 1:784]
rm(Test_0, Test_1, Test_2, Test_3, Test_4, Test_5, Test_6, Test_7, Test_8, Test_9); gc()

##*********************************
## 3. Componentes principales ####
##*********************************

## Aplicamos a continuación componentes principales. El objetivo será acelerar el cómputo del algoritmo, utilizando 
## una cantidad inferior de información conservando la mayor variabilidad posible de la información

### Centrado de datos
centro <- colMeans(train)
train <- train - matrix(centro, nrow(train), ncol(train), byrow=TRUE)
test <- test - matrix(centro, nrow(test), ncol(test), byrow=TRUE)

## Componentes principales
pc <- prcomp(train, center = F) 
pc_train <- pc$x
pc_test <- predict(pc, test)

## Observamos a continuación la varianza ganada con diferente cantidad de componentes principales
var_ganada <- cumsum(pc$sdev^2)/sum(pc$sdev^2)
plot(var_ganada)
abline(h=0.5, col="brown2")
abline(h=0.8, col="brown3")
abline(h=0.9, col="brown4")
abline(h=0.95, col="green4")
abline(h=0.99, col="green2")

which(var_ganada>0.5)[1] # 11 componentes
which(var_ganada>0.8)[1] # 44 componentes
which(var_ganada>0.9)[1] # 87 componentes
which(var_ganada>0.95)[1] # 154 componentes
which(var_ganada>0.99)[1] # 331 componentes


##*********************************
## 2. Red neuronal ####
##*********************************
set.seed(1013596884)

## Debido a que el resultado de nuestro modelo es multiclase, resulta necesario entrenas 10 modelos diferentes,
## aplicando un modelo 1 contra todos, es decir que para cada clase se compara contra las demás.
## Puesto que aplicamos el modelo 1 contra uno, es necesario balancear la muestra

## Se contruye el siguiente data frame, con la finalidad de identificar las posiciones correspondientes 
## a cada clase.
labels <- data.frame(ID=1:length(y_train), y=as.numeric(as.character(y_train)))
pos <- labels %>%
       group_by(y) %>%
       summarise(fila_desde = min(ID), fila_hasta = max(ID))

## A continuación se balnacea la muestra
vec_0 <- pos$fila_desde[1]:pos$fila_hasta[1]
vec_mues <- labels$ID[-(vec_0)]
sample_0 <- sample(vec_mues, size = length(vec_0))
y_0 <- ifelse(y_train[c(vec_0, sample_0)]=="0", 1, 0)
table(y_0)

vec_1 <- pos$fila_desde[2]:pos$fila_hasta[2]
vec_mues <- labels$ID[-(vec_1)]
sample_1 <- sample(vec_mues, size = length(vec_1))
y_1 <- ifelse(y_train[c(vec_1, sample_1)]=="1", 1, 0)
table(y_1)

vec_2 <- pos$fila_desde[3]:pos$fila_hasta[3]
vec_mues <- labels$ID[-(vec_2)]
sample_2 <- sample(vec_mues, size = length(vec_2))
y_2 <- ifelse(y_train[c(vec_2, sample_2)]=="2", 1, 0)
table(y_2)

vec_3 <- pos$fila_desde[4]:pos$fila_hasta[4]
vec_mues <- labels$ID[-(vec_3)]
sample_3 <- sample(vec_mues, size = length(vec_3))
y_3 <- ifelse(y_train[c(vec_3, sample_3)]=="3", 1, 0)
table(y_3)

vec_4 <- pos$fila_desde[5]:pos$fila_hasta[5]
vec_mues <- labels$ID[-(vec_4)]
sample_4 <- sample(vec_mues, size = length(vec_4))
y_4 <- ifelse(y_train[c(vec_4, sample_4)]=="4", 1, 0)
table(y_4)

vec_5 <- pos$fila_desde[6]:pos$fila_hasta[6]
vec_mues <- labels$ID[-(vec_5)]
sample_5 <- sample(vec_mues, size = length(vec_5))
y_5 <- ifelse(y_train[c(vec_5, sample_5)]=="5", 1, 0)
table(y_5)

vec_6 <- pos$fila_desde[7]:pos$fila_hasta[7]
vec_mues <- labels$ID[-(vec_6)]
sample_6 <- sample(vec_mues, size = length(vec_6))
y_6 <- ifelse(y_train[c(vec_6, sample_6)]=="6", 1, 0)
table(y_6)

vec_7 <- pos$fila_desde[8]:pos$fila_hasta[8]
vec_mues <- labels$ID[-(vec_7)]
sample_7 <- sample(vec_mues, size = length(vec_7))
y_7 <- ifelse(y_train[c(vec_7, sample_7)]=="7", 1, 0)
table(y_7)

vec_8 <- pos$fila_desde[9]:pos$fila_hasta[9]
vec_mues <- labels$ID[-(vec_8)]
sample_8 <- sample(vec_mues, size = length(vec_8))
y_8 <- ifelse(y_train[c(vec_8, sample_8)]=="8", 1, 0)
table(y_8)

vec_9 <- pos$fila_desde[10]:pos$fila_hasta[10]
vec_mues <- labels$ID[-(vec_9)]
sample_9 <- sample(vec_mues, size = length(vec_9))
y_9 <- ifelse(y_train[c(vec_9, sample_9)]=="9", 1, 0)
table(y_9)

## Componentes a usar
componentes <- 87 # >> 90% de la varianza

## Entrenamiento de la red neuronal para cada dígito. Debido al componente aleatorio contenido dentro de 
## las redes neuronales, optamos por replicar para cada dígito varias iteraciones con la función nnet.
## Los resultados finales se promediarán
iteraciones <- 1:5

net_0 <- list()
net_1 <- list()
net_2 <- list()
net_3 <- list()
net_4 <- list()
net_5 <- list()
net_6 <- list()
net_7 <- list()
net_8 <- list()
net_9 <- list()
require(progress)
pb <- progress_bar$new(total = length(iteraciones))
for(i in iteraciones){
     net_0[[i]] <- nnet(pc_train[c(vec_0, sample_0), 1:componentes], y_0, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_1[[i]] <- nnet(pc_train[c(vec_1, sample_1), 1:componentes], y_1, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_2[[i]] <- nnet(pc_train[c(vec_2, sample_2), 1:componentes], y_2, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_3[[i]] <- nnet(pc_train[c(vec_3, sample_3), 1:componentes], y_3, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_4[[i]] <- nnet(pc_train[c(vec_4, sample_4), 1:componentes], y_4, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_5[[i]] <- nnet(pc_train[c(vec_5, sample_5), 1:componentes], y_5, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_6[[i]] <- nnet(pc_train[c(vec_6, sample_6), 1:componentes], y_6, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_7[[i]] <- nnet(pc_train[c(vec_7, sample_7), 1:componentes], y_7, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_8[[i]] <- nnet(pc_train[c(vec_8, sample_8), 1:componentes], y_8, size=10, trace=F, maxit=500, MaxNWts=2000)
     net_9[[i]] <- nnet(pc_train[c(vec_9, sample_9), 1:componentes], y_9, size=10, trace=F, maxit=500, MaxNWts=2000)
     pb$tick()
}


## Guardamos los resultados
saveRDS(net_0, "results/net_0.rds")
saveRDS(net_1, "results/net_1.rds")
saveRDS(net_2, "results/net_2.rds")
saveRDS(net_3, "results/net_3.rds")
saveRDS(net_4, "results/net_4.rds")
saveRDS(net_5, "results/net_5.rds")
saveRDS(net_6, "results/net_6.rds")
saveRDS(net_7, "results/net_7.rds")
saveRDS(net_8, "results/net_8.rds")
saveRDS(net_9, "results/net_9.rds")

#********************************************
## 3. Resultados sobre conjunto prueba ####
#********************************************
## A continuación ejecutamos un algoritmo para calcular predicciones sobre los respectivos conjuntos de prueba.
## Se ejecuta una predicción por cada iteración, es decir, 5 predicciones por cada dígito.
pred_0 <- NULL
pred_1 <- NULL
pred_2 <- NULL
pred_3 <- NULL
pred_4 <- NULL
pred_5 <- NULL
pred_6 <- NULL
pred_7 <- NULL
pred_8 <- NULL
pred_9 <- NULL
for(i in iteraciones){
     pred_0[[i]] <- predict(net_0[[i]], pc_test[, 1:componentes])
     pred_1[[i]] <- predict(net_1[[i]], pc_test[, 1:componentes])
     pred_2[[i]] <- predict(net_2[[i]], pc_test[, 1:componentes])
     pred_3[[i]] <- predict(net_3[[i]], pc_test[, 1:componentes])
     pred_4[[i]] <- predict(net_4[[i]], pc_test[, 1:componentes])
     pred_5[[i]] <- predict(net_5[[i]], pc_test[, 1:componentes])
     pred_6[[i]] <- predict(net_6[[i]], pc_test[, 1:componentes])
     pred_7[[i]] <- predict(net_7[[i]], pc_test[, 1:componentes])
     pred_8[[i]] <- predict(net_8[[i]], pc_test[, 1:componentes])
     pred_9[[i]] <- predict(net_9[[i]], pc_test[, 1:componentes]) 
}


## Se define a continuación el resultado como el promedio de los resultados individuales de cada iteración, 
## para cada dígito
res_0 <- rowMeans(data.frame(pred_0[[1]], pred_0[[2]], pred_0[[3]], pred_0[[4]], pred_0[[5]]))
res_1 <- rowMeans(data.frame(pred_1[[1]], pred_1[[2]], pred_1[[3]], pred_1[[4]], pred_1[[5]]))
res_2 <- rowMeans(data.frame(pred_2[[1]], pred_2[[2]], pred_2[[3]], pred_2[[4]], pred_2[[5]]))
res_3 <- rowMeans(data.frame(pred_3[[1]], pred_3[[2]], pred_3[[3]], pred_3[[4]], pred_3[[5]]))
res_4 <- rowMeans(data.frame(pred_4[[1]], pred_4[[2]], pred_4[[3]], pred_4[[4]], pred_4[[5]]))
res_5 <- rowMeans(data.frame(pred_5[[1]], pred_5[[2]], pred_5[[3]], pred_5[[4]], pred_5[[5]]))
res_6 <- rowMeans(data.frame(pred_6[[1]], pred_6[[2]], pred_6[[3]], pred_6[[4]], pred_6[[5]]))
res_7 <- rowMeans(data.frame(pred_7[[1]], pred_7[[2]], pred_7[[3]], pred_7[[4]], pred_7[[5]]))
res_8 <- rowMeans(data.frame(pred_8[[1]], pred_8[[2]], pred_8[[3]], pred_8[[4]], pred_8[[5]]))
res_9 <- rowMeans(data.frame(pred_9[[1]], pred_9[[2]], pred_9[[3]], pred_9[[4]], pred_9[[5]]))


## La tabla a continuación define el resultado final como la máxima probabilidad enconttrada para 
## cada dígito
resultados <- data.frame(real_label=y_test, res_0, res_1, res_2, res_3, res_4, res_5, res_6,
                         res_7, res_8, res_9)
resultados$final_predict <- apply(resultados[,2:11], 1, which.max) - 1


## A continuación visualisamos el error de predicción  final sobre el conjunto de pruebas
## El error final es del 16.15%
error <- ifelse(resultados$real_label == resultados$final_predict, 0, 1)
sum(error)/length(error)


## Se genera función  para predecir la etiqueta para cualquier vector
predice_MNIST <- function(X){
     pred_0 <- NULL
     pred_1 <- NULL
     pred_2 <- NULL
     pred_3 <- NULL
     pred_4 <- NULL
     pred_5 <- NULL
     pred_6 <- NULL
     pred_7 <- NULL
     pred_8 <- NULL
     pred_9 <- NULL
     for(i in iteraciones){
          pred_0[[i]] <- predict(net_0[[i]], pc_test[X, 1:componentes])
          pred_1[[i]] <- predict(net_1[[i]], pc_test[X, 1:componentes])
          pred_2[[i]] <- predict(net_2[[i]], pc_test[X, 1:componentes])
          pred_3[[i]] <- predict(net_3[[i]], pc_test[X, 1:componentes])
          pred_4[[i]] <- predict(net_4[[i]], pc_test[X, 1:componentes])
          pred_5[[i]] <- predict(net_5[[i]], pc_test[X, 1:componentes])
          pred_6[[i]] <- predict(net_6[[i]], pc_test[X, 1:componentes])
          pred_7[[i]] <- predict(net_7[[i]], pc_test[X, 1:componentes])
          pred_8[[i]] <- predict(net_8[[i]], pc_test[X, 1:componentes])
          pred_9[[i]] <- predict(net_9[[i]], pc_test[X, 1:componentes]) 
     }
     
     res_0 <- rowMeans(data.frame(pred_0[[1]], pred_0[[2]], pred_0[[3]], pred_0[[4]], pred_0[[5]]))
     res_1 <- rowMeans(data.frame(pred_1[[1]], pred_1[[2]], pred_1[[3]], pred_1[[4]], pred_1[[5]]))
     res_2 <- rowMeans(data.frame(pred_2[[1]], pred_2[[2]], pred_2[[3]], pred_2[[4]], pred_2[[5]]))
     res_3 <- rowMeans(data.frame(pred_3[[1]], pred_3[[2]], pred_3[[3]], pred_3[[4]], pred_3[[5]]))
     res_4 <- rowMeans(data.frame(pred_4[[1]], pred_4[[2]], pred_4[[3]], pred_4[[4]], pred_4[[5]]))
     res_5 <- rowMeans(data.frame(pred_5[[1]], pred_5[[2]], pred_5[[3]], pred_5[[4]], pred_5[[5]]))
     res_6 <- rowMeans(data.frame(pred_6[[1]], pred_6[[2]], pred_6[[3]], pred_6[[4]], pred_6[[5]]))
     res_7 <- rowMeans(data.frame(pred_7[[1]], pred_7[[2]], pred_7[[3]], pred_7[[4]], pred_7[[5]]))
     res_8 <- rowMeans(data.frame(pred_8[[1]], pred_8[[2]], pred_8[[3]], pred_8[[4]], pred_8[[5]]))
     res_9 <- rowMeans(data.frame(pred_9[[1]], pred_9[[2]], pred_9[[3]], pred_9[[4]], pred_9[[5]]))
     
     resultados <- data.frame(real_label=y_test[X], res_0, res_1, res_2, res_3, res_4, res_5, res_6,
                              res_7, res_8, res_9)
     resultados$final_predict <- apply(resultados[,2:11], 1, which.max) - 1
     message("El número en esta imagen es... ", resultados$final_predict)
     return(resultados$final_predict)
}

## Ejecutamos la función anterior para un número cualquiera
predecir <- sample(1:nrow(test), 1)
# imagen que queremos predecir
show_digit(test[predecir, ])
## Reconocimiento
res_num <- predice_MNIST(predecir)

## Pasos a seguir:
## Debido a la carencia de tiempo a continuación se señalan algunas cosas adicionales que
## que podrían realizarse para continuar analizando resultados

## 1. Validación cruzada: Debido al componente aleatorio presente en las redes neuronales
##    esta validación cruzada permitiría minimizar el error por muestreo
## 2. Validaciones adicionales de los resultados: Visualización de matriz de confusión para
##    cada predicción de cada dpigito, precisión PPV, NPV, etc. 
## 3. Análisis gráfico: También es posible evaluar curva ROC para cada dígito.



