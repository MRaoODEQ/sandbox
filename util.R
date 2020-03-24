library(dplyr)
library(caret)
library(lubridate)
library(zeallot)
library(e1071)
library(pryr)
library(keras)
#install_keras()

sel <- read.csv("inputs/sel_multi.csv", stringsAsFactors = FALSE)


run_mlp <- function(df) {
    set.seed(23)
    trn0 <- df %>% filter(year(dtime) < 2015) %>% 
                   select(pm25_0, pm25_1, pm25_2, pm25_3, pm25_4, pm25_5, pm25_6,
                          hour, wday, month, temp, pr, ws, wd) %>%
                   filter(complete.cases(pm25_0, pm25_1, pm25_2, pm25_3, pm25_4, pm25_5, pm25_6,
                          hour, wday, month, temp, pr, ws, wd))
    valn0 <- df %>% filter(year(dtime) == 2015) %>% 
                    select(pm25_0, pm25_1, pm25_2, pm25_3, pm25_4, pm25_5, pm25_6,
                           hour, wday, month, temp, pr, ws, wd) %>%
                    filter(complete.cases(pm25_0, pm25_1, pm25_2, pm25_3, pm25_4, pm25_5, pm25_6,
                           hour, wday, month, temp, pr, ws, wd))
    preProcVal <- preProcess(trn0, method = c("center", "scale"))
    
    tr  <- predict(preProcVal, trn0)  %>% select(-c(pm25_0))
    val <- predict(preProcVal, valn0) %>% select(-c(pm25_0))
    tr_lab <- trn0 %>% select(pm25_0)
    val_lab <- valn0 %>% select(pm25_0)
    tr <- as.matrix(tr)
    val <- as.matrix(val)
    dimnames(tr) <- NULL
    dimnames(val) <- NULL
    tr_lab <- as.matrix(tr_lab)
    val_lab <- as.matrix(val_lab)
    dimnames(tr_lab) <- NULL
    dimnames(val_lab) <- NULL
    mlp <- keras_model_sequential()
    mlp %>% layer_dense(units = 48, activation = 'elu', input_shape = c(13)) %>%
      layer_dropout(rate = 0.4) %>%
      #            layer_dense(units = 32, activation = 'elu') %>%
      #            layer_dropout(rate = 0.4) %>%
      #            layer_dense(units = 32, activation = 'tanh') %>%
      #            layer_dropout(rate = 0.3) %>%
      #            layer_dense(units = 16, activation = 'linear') %>%
      #            layer_dropout(rate = 0.3) %>%
      layer_dense(units = 1) 
    
    
    mlp <- mlp %>% compile( loss = 'mse', optimizer = optimizer_nadam(), metrics = 'mae')
    hist <- mlp %>% fit(tr, tr_lab, epochs = 100, batch_size = 500, validation_split = 0.3)
    pred <- mlp %>% predict(val)
    return(list(mlp, val, pred))
}


c(mlpm, mlpm.01n, mlpm.05n, mlpm.1n, mlpm.2n, mlpm.5n, mlpm1n, mlpm5n, mlpm10n, mlpm20n) %<-% lapply(nr, run_mlp)


mlpn_mods <- list(mlpm, mlpm.01n, mlpm.05n, mlpm.1n, mlpm.2n, mlpm.5n, mlpm1n, mlpm5n, mlpm10n, mlpm20n)
mlpn_summ <- sapply(1:10, function(i) { postResample(mlpn_mods[[i]][3][[1]], val_lab)})

mlpn_pred <- sapply(1:10, function(i) { mlpn_mods[[i]][3][[1]]})
mlpn_pred <- cbind(val_lab, mlpn_pred)
colnames(mlpn_pred) <- c("pm25_0", "sel", "sel0.01","sel0.05", "sel0.10", "sel0.20",
                         "sel0.50", "sel1.00", "sel5.00", "sel10.0", "sel20.0")
write.csv(mlpn_pred, "./results/sel_noise_mlp.csv")



c(mlpm, mlpm.01d, mlpm.05d, mlpm.1d, mlpm.2d, mlpm.5d, mlpm1d, mlpm5d, mlpm10d, mlpm20d) %<-% lapply(dr, run_mlp)
mlpd_mods <- list(mlpm, mlpm.01d, mlpm.05d, mlpm.1d, mlpm.2d, mlpm.5d, mlpm1d, mlpm5d, mlpm10d, mlpm20d)
mlpd_summ <- sapply(1:10, function(i) { postResample(mlpd_mods[[i]][3][[1]], val_lab)})
mlpd_pred <- sapply(1:10, function(i) { mlpd_mods[[i]][3][[1]]})
mlpd_pred <- cbind(val_lab, mlpd_pred)
colnames(mlpd_pred) <- c("pm25_0", "sel", "sel0.01","sel0.05", "sel0.10", "sel0.20",
                                             "sel0.50", "sel1.00", "sel5.00", "sel10.0", "sel20.0")
write.csv(mlpd_pred, "./results/sel_drift_mlp.csv")


type <- "noise_mlp"
rownames(mlpn_summ) <- c(paste0(type, "_RMSE"), paste0(type, "_Rsquared"), paste0(type, "_MAE"))
type <- "drift_mlp"
rownames(mlpd_summ) <- c(paste0(type, "_RMSE"), paste0(type, "_Rsquared"), paste0(type, "_MAE"))
mlp_summ <- rbind(mlpn_summ, mlpd_summ)
colnames(mlp_summ) <- c("sel", "sel0.01","sel0.05", "sel0.10", "sel0.20",
                        "sel0.50", "sel1.00", "sel5.00", "sel10.0", "sel20.0")
write.csv(mlp_summ, "mlp_summ.csv")
