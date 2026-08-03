# ==========================================
# Models: 
#   1. LASSO  
#   2. SVM
# Feature Sets:
#   1. Unigrams + Topics
#   2. Topics Only (10 Dense Features)
#   3. Bag of Words (Unigrams)
#   4. Bag of Words (Bigrams)
#   5. Unigrams (TF-IDF Weighted)
# ==========================================

library(tm)
library(caret)
library(glmnet)
library(e1071)
library(NLP)
library(topicmodels)
library(slam)
library(textstem)
library(sentimentr)

if(!requireNamespace("textstem", quietly = TRUE)) {
  install.packages("textstem")
}
if(!requireNamespace("Lexicon", quietly = TRUE)) {
  install.packages("Lexicon")
}
if(!requireNamespace("sentimentr", quietly = TRUE)) {
  install.packages("sentimentr")
}

# 1. Load Data
# ------------------------------------------
root_folder <- "C:/Users/felle/Documents/Rafi/text_mining/text_mining_project"
transcripts_file_path <- file.path(root_folder, "transcripts_with_returns.csv")

df <- read.csv(transcripts_file_path, stringsAsFactors = FALSE)
df <- subset(df, `Is.Price.UP` %in% c("YES", "NO"))

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

# 3. Build Base DTMs & Safely Filter Empty Rows
# ------------------------------------------
dtm_unigram <- DocumentTermMatrix(corpus)

BigramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "))
}

dtm_bigram <- DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer))

categorial_param <- factor(df$`Is.Price.UP`, levels = c("NO", "YES"))

# 4. Construct Feature Sets
# ------------------------------------------
# Set A: Unigrams
dtm_uni_sparse <- removeSparseTerms(dtm_unigram, 0.95)
X_unigram      <- as.matrix(dtm_uni_sparse)

# Set B: Bigrams
dtm_bi_sparse <- removeSparseTerms(dtm_bigram, 0.95)
X_bigram      <- as.matrix(dtm_bi_sparse)

# Set C: LDA Topics (10 Topics)
lda_model  <- LDA(dtm_unigram, k = 10, method = "Gibbs", 
                  control = list(seed = 1, burnin = 500, iter = 1000))

X_topics_raw <- posterior(lda_model)$topics # Raw proportions (0.0 to 1.0)
colnames(X_topics_raw) <- paste0("Topic_", 1:10)

X_topics_scaled <- X_topics_raw

# Set D: Combined Unigrams + Raw Topic Proportions
X_unigram_topics <- cbind(X_unigram, X_topics_raw)

# Set E: TF-IDF (Built directly from row-filtered dtm_unigram)
dtm_tfidf        <- weightTfIdf(dtm_unigram)
dtm_tfidf_sparse <- removeSparseTerms(dtm_tfidf, 0.95)
X_tfidf          <- as.matrix(dtm_tfidf_sparse)

feature_sets <- list(
  "1. Bigrams (Bag of Words)"          = X_bigram,
  "2. Unigrams (Bag of Words)"         = X_unigram,
  "3. Topics Only" = X_topics_scaled,
  "4. Unigrams + Topic Modeling"       = X_unigram_topics,
  "5. Unigrams (TF-IDF Weighted)"      = X_tfidf
)

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
  run_classifiers(feature_sets[[name]], categorial_param, name)
}
