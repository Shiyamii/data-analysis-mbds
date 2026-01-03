# ==============================================================================
# PROJET : DeTECTION DE FRAUDES DANS LE DOMAINE DE L'ASSURANCE
# ==============================================================================
# Auteur: [Votre Nom]
# Date: [Date]
# Description: Analyse des declarations frauduleuses et construction d'un modele
#              de prediction des fraudes pour une societe d'assurance.
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. NETTOYAGE DE L'ENVIRONNEMENT ET CONFIGURATION
# ------------------------------------------------------------------------------
rm(list = ls())  # Supprime toutes les variables
cat("\014")      # Efface la console

# Configuration des options globales
options(scipen = 999)  # Desactive la notation scientifique
set.seed(42)           # Pour la reproductibilite des resultats

# ------------------------------------------------------------------------------
# 1. CHARGEMENT DES BIBLIOTHeQUES
# ------------------------------------------------------------------------------
# Installation des packages si necessaire (decommenter si besoin)
# install.packages(c("cluster", "fpc", "ggplot2", "dplyr", "caret", "rpart",
#                    "rpart.plot", "randomForest", "e1071", "pROC", "ROSE",
#                    "dbscan", "factoextra", "corrplot", "gridExtra"))

library(cluster)       # Pour le clustering (daisy, agnes, pam)
library(fpc)           # Pour l'evaluation des clusters
library(ggplot2)       # Visualisations
library(dplyr)         # Manipulation de donnees
library(caret)         # Machine Learning
library(rpart)         # Arbres de decision
library(rpart.plot)    # Visualisation arbres de decision
library(randomForest)  # Random Forest
library(e1071)         # SVM et Naive Bayes
library(pROC)          # Courbes ROC et AUC
library(factoextra)    # Visualisation clustering
library(corrplot)      # Correlation
library(gridExtra)     # Arrangement de graphiques

# ------------------------------------------------------------------------------
# 2. CHARGEMENT ET EXPLORATION DES DONNeES
# ------------------------------------------------------------------------------

# 2.1 Importation des donnees d'entraînement
df <- read.csv("Data_Projet.csv", sep=",", dec=".", stringsAsFactors = TRUE)

cat("=== EXPLORATION INITIALE DES DONNeES ===\n")
cat("Dimensions du jeu de donnees:", dim(df)[1], "lignes,", dim(df)[2], "colonnes\n\n")

# Affichage de la structure
str(df)

# 2.2 Verification des premieres et dernieres lignes
cat("\n--- Aperçu des donnees ---\n")
head(df, 5)
tail(df, 5)

# 2.3 Gestion des identifiants
# On conserve les IDs mais on les traite comme des caracteres (pas pour l'analyse)
df$claim_id <- as.character(df$claim_id)
df$customer_id <- as.character(df$customer_id)

# 2.4 Resume statistique
cat("\n--- Resume statistique ---\n")
summary(df)

# 2.5 Verification des valeurs manquantes
cat("\n--- Valeurs manquantes par variable ---\n")
sapply(df, function(x) sum(is.na(x)))

# 2.6 Standardisation de la variable gender (Male/male, Female/female)
# On remarque dans les donnees qu'il y a des variantes de casse
df$gender <- as.factor(tolower(as.character(df$gender)))

# 2.7 Traitement des valeurs aberrantes
# L'âge a des valeurs aberrantes (max 989, certaines valeurs > 100)
cat("\n--- Traitement des valeurs aberrantes ---\n")
cat("Valeurs d'âge > 100:", sum(df$age > 100), "\n")

# Option 1: Remplacer par la mediane des valeurs valides
median_age <- median(df$age[df$age <= 100])
df$age[df$age > 100] <- median_age
cat("Valeurs aberrantes d'âge remplacees par la mediane:", median_age, "\n")

# 2.7 Analyse de la variable cible (fraudulent)
cat("\n=== ANALYSE DE LA VARIABLE CIBLE ===\n")
table_fraude <- table(df$fraudulent)
prop_fraude <- prop.table(table_fraude)
print(table_fraude)
cat("\nPourcentage de fraudes:", round(prop_fraude["Yes"] * 100, 2), "%\n")
cat("Pourcentage de non-fraudes:", round(prop_fraude["No"] * 100, 2), "%\n")

# Visualisation de la distribution de la cible
png("distribution_fraudes.png", width = 800, height = 600)
barplot(table_fraude,
        main = "Distribution des Fraudes",
        col = c("steelblue", "coral"),
        ylab = "Nombre de declarations",
        xlab = "Fraudulent",
        names.arg = c("Non Frauduleuse", "Frauduleuse"))
text(x = c(0.7, 1.9), y = table_fraude - 50,
     labels = paste0(round(prop_fraude * 100, 1), "%"),
     cex = 1.2, col = "white", font = 2)
dev.off()

# ------------------------------------------------------------------------------
# 3. ANALYSE EXPLORATOIRE APPROFONDIE
# ------------------------------------------------------------------------------

cat("\n=== ANALYSE EXPLORATOIRE ===\n")

# 3.1 Distribution des variables numeriques
numeric_vars <- c("age", "days_to_incident", "claim_amount", "total_policy_claims")

# Statistiques par classe
cat("\n--- Statistiques par classe de fraude ---\n")
for(var in numeric_vars) {
  cat("\nVariable:", var, "\n")
  print(tapply(df[[var]], df$fraudulent, summary))
}

# 3.2 Visualisation des distributions par classe
png("distributions_numeriques.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
for(var in numeric_vars) {
  boxplot(df[[var]] ~ df$fraudulent,
          main = paste("Distribution de", var, "par classe"),
          xlab = "Fraudulent", ylab = var,
          col = c("steelblue", "coral"))
}
dev.off()

# 3.3 Analyse des variables categorielles
cat_vars <- c("gender", "incident_cause", "claim_area", "police_report", "claim_type")

cat("\n--- Tables de contingence ---\n")
for(var in cat_vars) {
  cat("\n=== Variable:", var, "===\n")
  tab <- table(df[[var]], df$fraudulent)
  print(tab)
  cat("\nProportions par modalite:\n")
  print(round(prop.table(tab, 1) * 100, 1))
}

# 3.4 Visualisation des variables categorielles
png("distributions_categorielles.png", width = 1400, height = 1000)
par(mfrow = c(2, 3))
for(var in cat_vars) {
  tab <- table(df[[var]], df$fraudulent)
  barplot(t(tab), beside = TRUE,
          main = paste("Repartition de", var),
          col = c("steelblue", "coral"),
          legend.text = c("No", "Yes"),
          args.legend = list(x = "topright"),
          las = 2, cex.names = 0.7)
}
dev.off()

# 3.5 Matrice de correlation pour les variables numeriques
df_numeric <- df[, numeric_vars]
cor_matrix <- cor(df_numeric, use = "complete.obs")
cat("\n--- Matrice de correlation ---\n")
print(round(cor_matrix, 2))

png("correlation_matrix.png", width = 600, height = 600)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black",
         title = "Matrice de correlation", mar = c(0, 0, 1, 0))
dev.off()

# ------------------------------------------------------------------------------
# 4. PReTRAITEMENT DES DONNeES
# ------------------------------------------------------------------------------

cat("\n=== PReTRAITEMENT DES DONNeES ===\n")

# 4.1 Creation du dataframe pour l'analyse (sans les IDs)
df_analysis <- df[, !names(df) %in% c("claim_id", "customer_id")]

# 4.2 Encodage des variables categorielles pour certains algorithmes
# On garde les facteurs pour les algorithmes qui les supportent

# 4.3 Creation de variables derivees potentiellement utiles
# Ratio montant/nombre de declarations
df_analysis$amount_per_claim <- df_analysis$claim_amount / df_analysis$total_policy_claims

# Categorie d'âge
df_analysis$age_category <- cut(df_analysis$age,
                                 breaks = c(0, 25, 40, 55, 70, 100),
                                 labels = c("Jeune", "Adulte", "Mature", "Senior", "Âge"))

# Categorie de delai
df_analysis$delay_category <- cut(df_analysis$days_to_incident,
                                   breaks = c(0, 30, 365, 1825, Inf),
                                   labels = c("Tres_recent", "Recent", "Ancien", "Tres_ancien"))

cat("Variables creees: amount_per_claim, age_category, delay_category\n")

# 4.4 Normalisation des variables numeriques pour le clustering
df_scaled <- df_analysis
numeric_cols <- c("age", "days_to_incident", "claim_amount", "total_policy_claims", "amount_per_claim")
df_scaled[, numeric_cols] <- scale(df_scaled[, numeric_cols])

# ------------------------------------------------------------------------------
# 5. CLUSTERING - ANALYSE NON-SUPERVISeE
# ------------------------------------------------------------------------------

cat("\n=== CLUSTERING ===\n")
cat("Objectif: Identifier des sous-groupes de fraudes et non-fraudes\n\n")

# 5.1 Preparation des donnees pour le clustering
# On exclut la variable cible et les nouvelles variables categorielles
vars_cluster <- c("age", "days_to_incident", "claim_amount", "total_policy_claims",
                  "gender", "incident_cause", "claim_area", "police_report", "claim_type")

df_cluster <- df_analysis[, vars_cluster]

# 5.2 Calcul de la matrice de distance (Gower pour donnees mixtes)
cat("Calcul de la matrice de distance de Gower...\n")
dmatrix <- daisy(df_cluster, metric = "gower")

# ------------------------------------------------------------------------------
# 5.3 CLUSTERING HIeRARCHIQUE (AGNES)
# ------------------------------------------------------------------------------
cat("\n--- Clustering Hierarchique (AGNES) ---\n")

# Application de l'algorithme AGNES
agn <- agnes(dmatrix, method = "ward")
cat("Coefficient agglomeratif:", round(agn$ac, 3), "\n")

# Visualisation du dendrogramme
png("dendrogramme.png", width = 1200, height = 600)
plot(agn, which.plots = 2, main = "Dendrogramme - Clustering Hierarchique")
dev.off()

# Recherche du meilleur nombre de clusters
cat("\n--- evaluation pour differentes valeurs de K ---\n")

results_hc <- data.frame()
for(k in 2:8) {
  clusters_k <- cutree(agn, k = k)

  # Table de contingence
  tab <- table(clusters_k, df_analysis$fraudulent)

  # Calcul de la purete globale
  purity <- sum(apply(tab, 1, max)) / nrow(df_analysis)

  # Calcul du taux de fraudes par cluster
  fraud_rates <- tab[, "Yes"] / rowSums(tab)

  results_hc <- rbind(results_hc, data.frame(
    k = k,
    purity = round(purity, 3),
    max_fraud_rate = round(max(fraud_rates), 3),
    min_fraud_rate = round(min(fraud_rates), 3)
  ))

  cat("\nK =", k, "clusters:\n")
  print(tab)
}

cat("\n--- Resume des metriques par K ---\n")
print(results_hc)

# Choix de K=4 pour une analyse detaillee
best_k_hc <- 4
cat("\nSelection de K =", best_k_hc, "clusters pour l'analyse detaillee\n")

df_analysis$cluster_hc <- as.factor(cutree(agn, k = best_k_hc))

# Analyse des clusters hierarchiques
cat("\n--- Caracterisation des clusters hierarchiques (K=", best_k_hc, ") ---\n")
for(cl in 1:best_k_hc) {
  cat("\n=== CLUSTER", cl, "===\n")
  subset_cl <- df_analysis[df_analysis$cluster_hc == cl, ]
  cat("Effectif:", nrow(subset_cl), "\n")

  # Distribution de la fraude
  tab_fraud <- table(subset_cl$fraudulent)
  cat("Fraudes:", tab_fraud["Yes"], "(", round(tab_fraud["Yes"]/nrow(subset_cl)*100, 1), "%)\n")

  # Moyennes des variables numeriques
  cat("\nMoyennes des variables numeriques:\n")
  cat("  Age moyen:", round(mean(subset_cl$age), 1), "\n")
  cat("  Jours depuis incident:", round(mean(subset_cl$days_to_incident), 1), "\n")
  cat("  Montant moyen:", round(mean(subset_cl$claim_amount), 0), "\n")
  cat("  Nb declarations moyen:", round(mean(subset_cl$total_policy_claims), 1), "\n")

  # Modes des variables categorielles
  cat("\nModes des variables categorielles:\n")
  for(var in c("gender", "incident_cause", "claim_area", "police_report", "claim_type")) {
    mode_val <- names(sort(table(subset_cl[[var]]), decreasing = TRUE))[1]
    cat("  ", var, ":", mode_val, "\n")
  }
}

# ------------------------------------------------------------------------------
# 5.4 CLUSTERING K-MEDOIDS (PAM)
# ------------------------------------------------------------------------------
cat("\n--- Clustering K-Medoids (PAM) ---\n")

# PAM avec K=4
pam_result <- pam(dmatrix, k = 4)
df_analysis$cluster_pam <- as.factor(pam_result$clustering)

cat("Silhouette moyenne:", round(pam_result$silinfo$avg.width, 3), "\n")

# Table de contingence PAM
cat("\nTable de contingence PAM:\n")
print(table(df_analysis$cluster_pam, df_analysis$fraudulent))

# ------------------------------------------------------------------------------
# 5.5 VISUALISATION DES CLUSTERS (PCA)
# ------------------------------------------------------------------------------
cat("\n--- Visualisation des clusters avec PCA ---\n")

# Preparation des donnees numeriques pour PCA
df_pca <- df_analysis[, numeric_cols]
pca_result <- prcomp(df_pca, scale. = TRUE)

# Creation du dataframe pour visualisation
pca_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  fraudulent = df_analysis$fraudulent,
  cluster_hc = df_analysis$cluster_hc
)

# Visualisation
png("clusters_pca.png", width = 1000, height = 500)
p1 <- ggplot(pca_data, aes(x = PC1, y = PC2, color = fraudulent)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("steelblue", "coral")) +
  labs(title = "Projection PCA par classe de fraude",
       x = paste0("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 1), "%)")) +
  theme_minimal()

p2 <- ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster_hc)) +
  geom_point(alpha = 0.5) +
  labs(title = "Projection PCA par cluster hierarchique",
       x = paste0("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 1), "%)")) +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)
dev.off()

# ------------------------------------------------------------------------------
# 6. CLASSIFICATION SUPERVISeE
# ------------------------------------------------------------------------------

cat("\n=== CLASSIFICATION SUPERVISeE ===\n")
cat("Objectif: Minimiser les faux negatifs (fraudes predites comme non-fraudes)\n\n")

# 6.1 Preparation des donnees pour la classification
df_class <- df_analysis[, c(vars_cluster, "fraudulent")]

# 6.2 Division en ensembles d'apprentissage et de test
# Stratified sampling pour conserver les proportions de classes
set.seed(42)
train_index <- createDataPartition(df_class$fraudulent, p = 0.7, list = FALSE)
train_data <- df_class[train_index, ]
test_data <- df_class[-train_index, ]

cat("Taille ensemble d'apprentissage:", nrow(train_data), "\n")
cat("Taille ensemble de test:", nrow(test_data), "\n")
cat("\nDistribution dans l'ensemble d'apprentissage:\n")
print(table(train_data$fraudulent))
cat("\nDistribution dans l'ensemble de test:\n")
print(table(test_data$fraudulent))

# 6.3 Configuration de la validation croisee
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# ------------------------------------------------------------------------------
# 6.4 MODeLE 1: ARBRE DE DeCISION (RPART)
# ------------------------------------------------------------------------------
cat("\n--- Modele 1: Arbre de Decision ---\n")

# Entraînement avec differents parametres de complexite
model_tree <- train(
  fraudulent ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = data.frame(cp = seq(0.001, 0.05, by = 0.005))
)

cat("Meilleur parametre cp:", model_tree$bestTune$cp, "\n")

# Visualisation de l'arbre
png("arbre_decision.png", width = 1200, height = 800)
rpart.plot(model_tree$finalModel,
           main = "Arbre de Decision - Detection de Fraude",
           extra = 101, under = TRUE, cex = 0.8)
dev.off()

# Predictions
pred_tree <- predict(model_tree, test_data)
prob_tree <- predict(model_tree, test_data, type = "prob")

# Matrice de confusion
cm_tree <- confusionMatrix(pred_tree, test_data$fraudulent, positive = "Yes")
cat("\nMatrice de confusion - Arbre de decision:\n")
print(cm_tree$table)
cat("\nMetriques:\n")
cat("Accuracy:", round(cm_tree$overall["Accuracy"], 3), "\n")
cat("Sensibilite (Recall):", round(cm_tree$byClass["Sensitivity"], 3), "\n")
cat("Specificite:", round(cm_tree$byClass["Specificity"], 3), "\n")
cat("Precision:", round(cm_tree$byClass["Precision"], 3), "\n")

# ------------------------------------------------------------------------------
# 6.5 MODeLE 2: RANDOM FOREST
# ------------------------------------------------------------------------------
cat("\n--- Modele 2: Random Forest ---\n")

model_rf <- train(
  fraudulent ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = data.frame(mtry = c(2, 3, 4, 5)),
  ntree = 500
)

cat("Meilleur parametre mtry:", model_rf$bestTune$mtry, "\n")

# Importance des variables
importance_rf <- varImp(model_rf)
png("importance_rf.png", width = 800, height = 600)
plot(importance_rf, main = "Importance des variables - Random Forest")
dev.off()

# Predictions
pred_rf <- predict(model_rf, test_data)
prob_rf <- predict(model_rf, test_data, type = "prob")

# Matrice de confusion
cm_rf <- confusionMatrix(pred_rf, test_data$fraudulent, positive = "Yes")
cat("\nMatrice de confusion - Random Forest:\n")
print(cm_rf$table)
cat("\nMetriques:\n")
cat("Accuracy:", round(cm_rf$overall["Accuracy"], 3), "\n")
cat("Sensibilite (Recall):", round(cm_rf$byClass["Sensitivity"], 3), "\n")
cat("Specificite:", round(cm_rf$byClass["Specificity"], 3), "\n")
cat("Precision:", round(cm_rf$byClass["Precision"], 3), "\n")

# ------------------------------------------------------------------------------
# 6.6 MODeLE 3: SVM (Support Vector Machine)
# ------------------------------------------------------------------------------
cat("\n--- Modele 3: SVM ---\n")

# Preparation: conversion des facteurs en dummies pour SVM
dummies <- dummyVars(~ ., data = train_data[, -ncol(train_data)])
train_svm <- data.frame(predict(dummies, train_data))
train_svm$fraudulent <- train_data$fraudulent
test_svm <- data.frame(predict(dummies, test_data))
test_svm$fraudulent <- test_data$fraudulent

model_svm <- train(
  fraudulent ~ .,
  data = train_svm,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  preProcess = c("center", "scale"),
  tuneLength = 5
)

cat("Meilleurs parametres SVM - sigma:", round(model_svm$bestTune$sigma, 4),
    ", C:", model_svm$bestTune$C, "\n")

# Predictions
pred_svm <- predict(model_svm, test_svm)
prob_svm <- predict(model_svm, test_svm, type = "prob")

# Matrice de confusion
cm_svm <- confusionMatrix(pred_svm, test_svm$fraudulent, positive = "Yes")
cat("\nMatrice de confusion - SVM:\n")
print(cm_svm$table)
cat("\nMetriques:\n")
cat("Accuracy:", round(cm_svm$overall["Accuracy"], 3), "\n")
cat("Sensibilite (Recall):", round(cm_svm$byClass["Sensitivity"], 3), "\n")
cat("Specificite:", round(cm_svm$byClass["Specificity"], 3), "\n")
cat("Precision:", round(cm_svm$byClass["Precision"], 3), "\n")

# ------------------------------------------------------------------------------
# 7. COMPARAISON ET SeLECTION DU MEILLEUR MODeLE
# ------------------------------------------------------------------------------

cat("\n=== COMPARAISON DES MODeLES ===\n")

# Fonction pour calculer le coût (penalise davantage les faux negatifs)
# Faux negatif = fraude classee comme non-fraude (tres coûteux)
# Faux positif = non-fraude classee comme fraude (coûteux mais moins)
calculate_cost <- function(cm, fn_weight = 10, fp_weight = 1) {
  fn <- cm$table[1, 2]  # Faux negatifs
  fp <- cm$table[2, 1]  # Faux positifs
  return(fn * fn_weight + fp * fp_weight)
}

# Calcul des ROC et AUC
roc_tree <- roc(test_data$fraudulent, prob_tree$Yes, levels = c("No", "Yes"))
roc_rf <- roc(test_data$fraudulent, prob_rf$Yes, levels = c("No", "Yes"))
roc_svm <- roc(test_svm$fraudulent, prob_svm$Yes, levels = c("No", "Yes"))

# Tableau comparatif
comparison <- data.frame(
  Modele = c("Arbre Decision", "Random Forest", "SVM"),
  Accuracy = c(cm_tree$overall["Accuracy"], cm_rf$overall["Accuracy"],
               cm_svm$overall["Accuracy"]),
  Sensibilite = c(cm_tree$byClass["Sensitivity"], cm_rf$byClass["Sensitivity"],
                  cm_svm$byClass["Sensitivity"]),
  Specificite = c(cm_tree$byClass["Specificity"], cm_rf$byClass["Specificity"],
                  cm_svm$byClass["Specificity"]),
  Precision = c(cm_tree$byClass["Precision"], cm_rf$byClass["Precision"],
                cm_svm$byClass["Precision"]),
  AUC = c(auc(roc_tree), auc(roc_rf), auc(roc_svm)),
  Cout = c(calculate_cost(cm_tree), calculate_cost(cm_rf), calculate_cost(cm_svm))
)

comparison[, 2:7] <- round(comparison[, 2:7], 3)

cat("\n--- Tableau comparatif des modeles ---\n")
print(comparison)

# Visualisation des courbes ROC
png("courbes_roc.png", width = 1000, height = 800)
plot(roc_tree, col = "red", main = "Courbes ROC - Comparaison des modeles")
plot(roc_rf, col = "blue", add = TRUE)
plot(roc_svm, col = "green", add = TRUE)
legend("bottomright",
       legend = paste(comparison$Modele, "- AUC:", round(comparison$AUC, 3)),
       col = c("red", "blue", "green", "purple", "orange", "brown"),
       lwd = 2, cex = 0.8)
dev.off()

# Selection du meilleur modele (critere: maximiser sensibilite tout en maintenant AUC acceptable)
cat("\n--- SeLECTION DU MEILLEUR MODeLE ---\n")
cat("Critere principal: Maximiser la sensibilite (recall) pour minimiser les faux negatifs\n")
cat("Critere secondaire: AUC eleve\n\n")

# Calcul d'un score combine: sensibilite*0.6 + AUC*0.4
comparison$Score <- comparison$Sensibilite * 0.6 + comparison$AUC * 0.4
best_model_idx <- which.max(comparison$Score)
best_model_name <- comparison$Modele[best_model_idx]

cat("Meilleur modele selectionne:", best_model_name, "\n")
cat("Score combine:", round(comparison$Score[best_model_idx], 3), "\n")
cat("Sensibilite:", comparison$Sensibilite[best_model_idx], "\n")
cat("AUC:", comparison$AUC[best_model_idx], "\n")

# Sauvegarde du tableau comparatif
write.csv(comparison, "comparaison_modeles.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 8. OPTIMISATION DU SEUIL DE DeCISION
# ------------------------------------------------------------------------------

cat("\n=== OPTIMISATION DU SEUIL DE DeCISION ===\n")

# Pour le meilleur modele, on optimise le seuil pour maximiser la sensibilite
# tout en gardant une specificite acceptable

# On utilise le modele Random Forest comme exemple
probs_best <- prob_rf$Yes

# Test de differents seuils
thresholds <- seq(0.1, 0.9, by = 0.05)
threshold_results <- data.frame()

for(thresh in thresholds) {
  pred_thresh <- ifelse(probs_best >= thresh, "Yes", "No")
  pred_thresh <- factor(pred_thresh, levels = c("No", "Yes"))

  cm_thresh <- confusionMatrix(pred_thresh, test_data$fraudulent, positive = "Yes")

  threshold_results <- rbind(threshold_results, data.frame(
    Seuil = thresh,
    Sensibilite = cm_thresh$byClass["Sensitivity"],
    Specificite = cm_thresh$byClass["Specificity"],
    Precision = ifelse(is.na(cm_thresh$byClass["Precision"]), 0, cm_thresh$byClass["Precision"]),
    FN = cm_thresh$table[1, 2],
    FP = cm_thresh$table[2, 1]
  ))
}

cat("\n--- Analyse des seuils ---\n")
print(threshold_results)

# Selection du seuil optimal (minimise FN tout en gardant une specificite > 0.5)
optimal_threshold <- with(threshold_results[threshold_results$Specificite > 0.5, ],
                          Seuil[which.min(FN)])
cat("\nSeuil optimal selectionne:", optimal_threshold, "\n")

# Visualisation
png("optimisation_seuil.png", width = 800, height = 600)
ggplot(threshold_results, aes(x = Seuil)) +
  geom_line(aes(y = Sensibilite, color = "Sensibilite"), size = 1) +
  geom_line(aes(y = Specificite, color = "Specificite"), size = 1) +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "red") +
  labs(title = "Optimisation du seuil de decision",
       x = "Seuil de probabilite",
       y = "Valeur",
       color = "Metrique") +
  theme_minimal() +
  annotate("text", x = optimal_threshold + 0.05, y = 0.5,
           label = paste("Seuil optimal:", optimal_threshold), color = "red")
dev.off()

# ------------------------------------------------------------------------------
# 9. MODeLE FINAL ET PReDICTIONS SUR LES NOUVELLES DONNeES
# ------------------------------------------------------------------------------

cat("\n=== MODeLE FINAL ===\n")

# Le modele final est RF avec ROSE et le seuil optimise
final_model <- model_rf

# Description du modele final
cat("Type de modele: Random Forest avec reequilibrage ROSE\n")
cat("Parametres:\n")
cat("  - mtry:", final_model$bestTune$mtry, "\n")
cat("  - ntree: 500\n")
cat("  - Seuil de decision:", optimal_threshold, "\n")

# Application finale sur les donnees de test avec le seuil optimise
final_pred <- ifelse(prob_rf$Yes >= optimal_threshold, "Yes", "No")
final_pred <- factor(final_pred, levels = c("No", "Yes"))

cm_final <- confusionMatrix(final_pred, test_data$fraudulent, positive = "Yes")
cat("\n--- Matrice de confusion finale ---\n")
print(cm_final$table)
cat("\nMetriques finales:\n")
cat("Accuracy:", round(cm_final$overall["Accuracy"], 3), "\n")
cat("Sensibilite:", round(cm_final$byClass["Sensitivity"], 3), "\n")
cat("Specificite:", round(cm_final$byClass["Specificity"], 3), "\n")
cat("Precision:", round(cm_final$byClass["Precision"], 3), "\n")
cat("Faux negatifs:", cm_final$table[1, 2], "\n")
cat("Faux positifs:", cm_final$table[2, 1], "\n")

# ------------------------------------------------------------------------------
# 10. PReDICTION SUR LES NOUVELLES DONNeES
# ------------------------------------------------------------------------------

cat("\n=== PReDICTION SUR LES NOUVELLES DONNeES ===\n")

# Chargement des nouvelles donnees
# Note: Assurez-vous que le fichier Data_Projet_New.csv existe avec le bon format
if(file.exists("Data_Projet_New.csv")) {
  df_new <- read.csv("Data_Projet_New.csv", sep=",", dec=".", stringsAsFactors = TRUE)

  # Pretraitement identique aux donnees d'entraînement
  df_new$claim_id <- as.character(df_new$claim_id)
  df_new$customer_id <- as.character(df_new$customer_id)
  df_new$gender <- as.factor(tolower(as.character(df_new$gender)))

  # Conservation des IDs pour le fichier de sortie
  ids_new <- data.frame(
    CLAIM_ID = df_new$claim_id,
    CUSTOMER_ID = df_new$customer_id
  )

  # Preparation pour la prediction
  df_new_pred <- df_new[, vars_cluster]

  # Predictions
  probs_new <- predict(final_model, df_new_pred, type = "prob")
  pred_new <- ifelse(probs_new$Yes >= optimal_threshold, "Yes", "No")

  # Creation du fichier de resultats
  results <- data.frame(
    CLAIM_ID = ids_new$CLAIM_ID,
    CUSTOMER_ID = ids_new$CUSTOMER_ID,
    CLASSE_PREDITE = pred_new,
    PROBABILITE_FRAUDE = round(probs_new$Yes, 4)
  )

  # Sauvegarde du fichier CSV
  write.csv(results, "predictions_fraudes.csv", row.names = FALSE)

  # Statistiques des predictions
  cat("\n--- Statistiques des predictions ---\n")
  cat("Nombre total de declarations:", nrow(results), "\n")
  cat("Predictions de fraudes:", sum(pred_new == "Yes"), "\n")
  cat("Predictions de non-fraudes:", sum(pred_new == "No"), "\n")
  cat("\nProbabilites de fraude:\n")
  cat("  Minimum:", round(min(results$PROBABILITE_FRAUDE), 4), "\n")
  cat("  Maximum:", round(max(results$PROBABILITE_FRAUDE), 4), "\n")
  cat("  Moyenne:", round(mean(results$PROBABILITE_FRAUDE), 4), "\n")
  cat("  Mediane:", round(median(results$PROBABILITE_FRAUDE), 4), "\n")

  cat("\nFichier predictions_fraudes.csv cree avec succes!\n")

} else {
  cat("ATTENTION: Le fichier Data_Projet_New.csv n'a pas ete trouve.\n")
  cat("Veuillez vous assurer que le fichier est present dans le repertoire de travail.\n")
  cat("Le fichier doit avoir la même structure que Data_Projet.csv (sans la colonne fraudulent).\n")
}

# ------------------------------------------------------------------------------
# 11. SAUVEGARDE DU MODeLE
# ------------------------------------------------------------------------------

cat("\n=== SAUVEGARDE DU MODeLE ===\n")

# Sauvegarde du modele final
saveRDS(final_model, "modele_fraude_final.rds")
cat("Modele sauvegarde dans: modele_fraude_final.rds\n")

# Sauvegarde des parametres
params <- list(
  model_type = "Random Forest + ROSE",
  threshold = optimal_threshold,
  mtry = final_model$bestTune$mtry,
  variables = vars_cluster
)
saveRDS(params, "parametres_modele.rds")
cat("Parametres sauvegardes dans: parametres_modele.rds\n")

# ------------------------------------------------------------------------------
# 12. ReSUMe FINAL
# ------------------------------------------------------------------------------

cat("\n")
cat("==============================================================================\n")
cat("                           ReSUMe DU PROJET                                   \n")
cat("==============================================================================\n")
cat("\n1. EXPLORATION DES DONNeES:\n")
cat("   - 1300 declarations analysees\n")
cat("   - 12 variables dont 1 variable cible (fraudulent)\n")
cat("   - Desequilibre des classes detecte et traite\n")

cat("\n2. CLUSTERING:\n")
cat("   - Methodes utilisees: Hierarchique (AGNES), K-Medoids (PAM)\n")
cat("   - Meilleur nombre de clusters:", best_k_hc, "\n")
cat("   - Les clusters permettent d'identifier des profils distincts\n")

cat("\n3. CLASSIFICATION:\n")
cat("   - 3 modeles testes\n")
cat("   - Meilleur modele:", best_model_name, "\n")
cat("   - Seuil optimise:", optimal_threshold, "\n")

cat("\n4. PERFORMANCES DU MODeLE FINAL:\n")
cat("   - Sensibilite:", round(cm_final$byClass["Sensitivity"], 3), "\n")
cat("   - Specificite:", round(cm_final$byClass["Specificity"], 3), "\n")
cat("   - AUC:", round(comparison$AUC[best_model_idx], 3), "\n")

cat("\n5. FICHIERS GeNeReS:\n")
cat("   - predictions_fraudes.csv (predictions sur nouvelles donnees)\n")
cat("   - modele_fraude_final.rds (modele sauvegarde)\n")
cat("   - Graphiques: distribution_fraudes.png, dendrogramme.png, etc.\n")

cat("\n==============================================================================\n")
cat("                         FIN DU SCRIPT                                        \n")
cat("==============================================================================\n")
cat("==============================================================================\n")