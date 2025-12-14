
# ----------------------------------------------
# 1. Chargement des bibliothèques nécessaires / nettoyage de l'environnement
# ----------------------------------------------

# 1.1. Importation des données
# On utilise stringsAsFactors = TRUE pour convertir directement les textes en catégories
df <- read.csv("Data_Projet.csv", sep=",", dec=".", stringsAsFactors = TRUE)

# 1.2. Vérification de la structure
dim(df)       # Doit afficher 1300 12
str(df)       # Affiche le type de chaque variable

# 1.3. Gestion des IDs
# On les retire des variables actives pour l'analyse, mais on les garde si besoin
# Ou on peut simplement les ignorer lors de la modélisation
df$claim_id <- as.character(df$claim_id)
df$customer_id <- as.character(df$customer_id)

# 1.4. Analyse de la variable cible (fraudulent)
table_fraude <- table(df$fraudulent)
prop_fraude <- prop.table(table_fraude)

print(table_fraude)
print(paste("Pourcentage de fraudes :", round(prop_fraude["Yes"] * 100, 2), "%"))

# Visualisation rapide de la cible
barplot(table_fraude, main="Distribution des Fraudes", col=c("green", "red"))

# 1.5. Résumé statistique des autres variables
print("Résumé statistique des variables :")
summary(df)


# ----------------------------------------------
# 2. Clustering
# ----------------------------------------------

# ----------------------------------------------
# 2.1 Clustering hiérarchique par agglomération
# ----------------------------------------------

# Installation des packages nécessaires si vous ne les avez pas
# install.packages(c("cluster", "fpc"))
library(cluster)
library(fpc)
library(tsne)

# 1. Préparation : On garde uniquement les variables explicatives
# On exclut les IDs (col 1 et 2) et la cible fraudulent (col 12 généralement, vérifiez l'index !)
# Supposons que fraudulent est la dernière colonne.
data_cluster <- df[, -c(1, 2, which(names(df) == "fraudulent"))]

# 2. Calcul de la matrice de distance
# Cela peut prendre quelques secondes
dmatrix <- daisy(data_cluster)

# 3. Clustering hiérarchique agglomératif (AGNES)
agn <- agnes(dmatrix)

# 4. Recherche du meilleur k avec une boucle
# On teste de 2 à 12 clusters
list_results <- list()

for(i in 2:12){
  print(paste("Calcul pour k =", i))
  # Resultats du clustering pour k=i
  agn_x <- cutree(agn, k = i)
  list_results[[as.character(i)]] <- agn_x

  # Créer la table de contingence affichant
  tab_cont <- table(agn_x, df$fraudulent)

  # Ajouter les totaux, la classe dominante et la pureté
  result <- data.frame(
    cluster_n = rownames(tab_cont),
    total = rowSums(tab_cont),
    classe_dominante = apply(tab_cont, 1, function(x) colnames(tab_cont)[which.max(x)]),
    purete_pct = round(apply(tab_cont, 1, max) / rowSums(tab_cont) * 100, 2),
    row.names = NULL
  )

  print(result)

  # Enregister l'histogramme d'effectifs
  hist_dir <- "searck_k"
  if (!dir.exists(hist_dir)) {
    dir.create(hist_dir)
  }
  plot_x <- qplot(agn_x, geom="bar", main=paste("Histogramme des clusters pour k =", i),
        xlab="Clusters", ylab="Effectifs", fill=I("blue"), col=I("black"))

  ggsave(
    filename = paste0(hist_dir,"/histogramme_clusters_k_", i, ".png"),
    plot = plot_x,
    width = 8,
    height = 6,
    dpi = 300
  )


}


# --- INTERPRÉTATION ---
# Regardez le graphique généré ci-dessus. Quel est le point le plus haut ?
# Supposons que le meilleur k soit 3 (à adapter selon votre graphique).

best_k <- 3

# 5. Application du Clustering final (Agglomération)
hc_final <- cutree(agn, k = best_k)
df$Cluster_HC <- as.factor(hc_final)
print(table(df$Cluster_HC))