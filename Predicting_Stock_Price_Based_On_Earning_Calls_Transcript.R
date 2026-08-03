# ==========================================
# Fixed Comprehensive Comparison Script
# Models: 
#   1. LASSO  
#   2. SVM
# Feature Sets:
#   1. Unigrams + Topics
#   2. Topics Only (10 Dense Features)
#   3. Bag of Words (Unigrams)
#   4. Bag of Words (Bigrams, Pruned)
#   5. Unigrams (TF-IDF Weighted)
# ==========================================

library(tm)
library(caret)
library(glmnet)
library(e1071)
library(NLP)
library(topicmodels)
library(slam)

# Install textstem (and its underlying lexicon package) if you haven't already
if(!requireNamespace("textstem", quietly = TRUE)) {
  install.packages("textstem")
}
if(!requireNamespace("Lexicon", quietly = TRUE)) {
  install.packages("Lexicon")
}

# Load the library at the top of your R script
library(textstem)

# ------------------------------------------
# 1. Load Data
# ------------------------------------------
root_folder <- "C:/Users/felle/Documents/Rafi/text_mining/text_mining_project"
transcripts_file_path <- file.path(root_folder, "transcripts_with_returns.csv")

df <- read.csv(transcripts_file_path, stringsAsFactors = FALSE)
df <- subset(df, `Is.Price.UP` %in% c("YES", "NO"))

# ------------------------------------------
# 2. Text Preprocessing Function
# ------------------------------------------
preprocess <- function(text){
  # Remove everything before the first occurrence of "Operator" (case-insensitive)
  # Before the word "Operator", the transcript includes just general technical information
  text <- sub("(?i)^.*?(?=operator)", "", text, perl = TRUE)
  
  myCorpus <- VCorpus(VectorSource(text))
  myCorpus <- tm_map(myCorpus, content_transformer(tolower))
  myCorpus <- tm_map(myCorpus, removeWords, stopwords("english"))
  myCorpus <- tm_map(myCorpus, removeNumbers)
  myCorpus <- tm_map(myCorpus, removePunctuation)
  myCorpus <- tm_map(myCorpus, stripWhitespace)
  myCorpus <- tm_map(myCorpus, content_transformer(lemmatize_strings))
  return(myCorpus)
}

corpus <- preprocess(df$TXT)

# ------------------------------------------
# 3. Build Base DTMs & Safely Filter Empty Rows
# ------------------------------------------
dtm_uni <- DocumentTermMatrix(corpus)

BigramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
}
dtm_bi <- DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer))

# Drop empty documents using sparse row sums
row_counts <- slam::row_sums(dtm_uni)
dtm_uni    <- dtm_uni[row_counts > 0, ]
dtm_bi     <- dtm_bi[row_counts > 0, ]
df         <- df[row_counts > 0, ]

y <- factor(df$`Is.Price.UP`, levels = c("NO", "YES"))

# ------------------------------------------
# 4. Construct Feature Sets
# ------------------------------------------
# Set A: Unigrams (Pruned to 0.95 sparsity)
dtm_uni_sparse <- removeSparseTerms(dtm_uni, 0.95)
X_unigram      <- as.matrix(dtm_uni_sparse)

# Set B: Bigrams (Pruned to 0.95 sparsity)
dtm_bi_sparse <- removeSparseTerms(dtm_bi, 0.95)
X_bigram      <- as.matrix(dtm_bi_sparse)

# Set C: LDA Topics (10 Dense Features)
num_topics <- 10
lda_model  <- LDA(dtm_uni, k = num_topics, method = "Gibbs", 
                  control = list(seed = 1, burnin = 500, iter = 1000))

X_topics_raw <- posterior(lda_model)$topics # Raw proportions (0.0 to 1.0)
colnames(X_topics_raw) <- paste0("Topic_", 1:num_topics)

X_topics_scaled <- X_topics_raw

# Set D: Combined Unigrams + Raw Topic Proportions
X_unigram_topics <- cbind(X_unigram, X_topics_raw)

# Set E: TF-IDF (Built directly from row-filtered dtm_uni)
dtm_tfidf        <- weightTfIdf(dtm_uni)
dtm_tfidf_sparse <- removeSparseTerms(dtm_tfidf, 0.95)
X_tfidf          <- as.matrix(dtm_tfidf_sparse)

feature_sets <- list(
  "1. Unigrams + Topic Modeling"       = X_unigram_topics,
  "2. Topics Only (10 Dense Features)" = X_topics_scaled,
  "3. Unigrams (Bag of Words)"         = X_unigram,
  "4. Bigrams (Bag of Words)"          = X_bigram,
  "5. Unigrams (TF-IDF Weighted)"      = X_tfidf
)

# ------------------------------------------
# 5. Model Execution Loop
# ------------------------------------------
run_classifiers <- function(X_matrix, y_vector, feature_set_name) {
  cat("\n=======================================================\n")
  cat(" FEATURE SET:", feature_set_name, "\n")
  cat(" Total Dimensions:", ncol(X_matrix), "features\n")
  cat("=======================================================\n")
  
  set.seed(1)
  train_index <- createDataPartition(y_vector, p = 0.6, list = FALSE)
  
  X_train <- X_matrix[train_index, ]
  X_test  <- X_matrix[-train_index, ]
  y_train <- y_vector[train_index]
  y_test  <- y_vector[-train_index]
  
  # --- A. LASSO Logistic Regression ---
  cv_model          <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1)
  lasso_pred        <- predict(cv_model, newx = X_test, s = "lambda.min", type = "class")
  lasso_pred_factor <- factor(as.character(lasso_pred), levels = c("NO", "YES"))
  
  cat("\n--- LASSO Logistic Regression ---\n")
  print(confusionMatrix(lasso_pred_factor, y_test))
  
  # --- B. SVM ---
  svm             <- svm(x = X_train, y = y_train, gamma = 0.1)
  svm_pred <- predict(svm, X_test)
  
  cat("\n--- SVM ---\n")
  print(confusionMatrix(svm_pred, y_test))
}

for (name in names(feature_sets)) {
  run_classifiers(feature_sets[[name]], y, name)
}
