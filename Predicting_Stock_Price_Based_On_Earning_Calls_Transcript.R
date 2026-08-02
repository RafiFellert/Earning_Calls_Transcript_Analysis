# Required Libraries
if(!requireNamespace("readr", quietly = TRUE)) install.packages("readr")
if(!requireNamespace("tm", quietly = TRUE)) install.packages("tm")
if(!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
if(!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
if(!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if(!requireNamespace("SnowballC", quietly = TRUE)) install.packages("SnowballC")

library(readr)
library(tm)
library(glmnet)
library(e1071)
library(caret)

options(stringsAsFactors = FALSE)
Sys.setlocale('LC_ALL', 'C')

# ==========================================
# 1. LOAD DATA & CLEAN TARGET VARIABLE
# ==========================================
# Updated to the path where your CSV was saved
csv_path <- "C:/Users/felle/transcripts_with_returns.csv"
df <- read_csv(csv_path)

# Filter out rows with missing or invalid market data if any
df <- df[df$`Is Price UP` %in% c("YES", "NO"), ]

# Convert target to a factor (1 for YES, 0 for NO)
df$Target_Factor <- factor(df$`Is Price UP`, levels = c("NO", "YES"))
df$Target_Num    <- ifelse(df$`Is Price UP` == "YES", 1, 0)


# ==========================================
# 2. TEXT PREPROCESSING & DTM CREATION
# ==========================================
preprocess_corpus <- function(text_vector) {
  corpus <- VCorpus(VectorSource(text_vector))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}

corpus <- preprocess_corpus(df$TXT)
dtm    <- DocumentTermMatrix(corpus)

# Remove extremely sparse terms (keep terms appearing in at least 10% of documents)
dtm <- removeSparseTerms(dtm, 0.90)

# Convert DTM to matrix and tf-idf weights (or term frequencies)
X <- as.matrix(dtm)
y_factor <- df$Target_Factor
y_num    <- df$Target_Num

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================
set.seed(42)
trainIndex <- createDataPartition(y_factor, p = 0.7, list = FALSE)

X_train <- X[trainIndex, ]
y_train_num <- y_num[trainIndex]
y_train_factor <- y_factor[trainIndex]

X_test <- X[-trainIndex, ]
y_test_num <- y_num[-trainIndex]
y_test_factor <- y_factor[-trainIndex]

# ==========================================
# 4. METHOD 1: LASSO LOGISTIC REGRESSION
# ==========================================
# alpha = 1 specifies Lasso (L1 penalty)
# Uses 5-fold cross-validation to find the optimal tuning parameter lambda
cv_lasso <- cv.glmnet(X_train, y_train_num, family = "binomial", alpha = 1, nfolds = 5)

# Predict on test set using optimal lambda
lasso_probs <- predict(cv_lasso, newx = X_test, s = "lambda.min", type = "response")
lasso_preds <- factor(ifelse(lasso_probs > 0.5, "YES", "NO"), levels = c("NO", "YES"))

cat("\n--- LASSO MODEL EVALUATION ---\n")
print(confusionMatrix(lasso_preds, y_test_factor))

# Extract non-zero Lasso coefficients (Top Predictive Words)
lasso_coefs <- coef(cv_lasso, s = "lambda.min")
active_words <- data.frame(
  Word = rownames(lasso_coefs)[lasso_coefs[,1] != 0],
  Coefficient = lasso_coefs[lasso_coefs[,1] != 0, 1]
)
print("Words selected by Lasso:")
print(active_words)

# ==========================================
# 5. METHOD 2: SUPPORT VECTOR MACHINE (SVM)
# ==========================================
# Option A: SVM with linear kernel
svm_model <- svm(x = X_train, y = y_train_factor, kernel = "linear", cost = 1)

svm_preds <- predict(svm_model, newdata = X_test)

cat("\n--- SVM MODEL EVALUATION ---\n")
print(confusionMatrix(svm_preds, y_test_factor))

# ==========================================
# 6. METHOD 3: HYBRID LASSO-SVM (FEATURE SELECTION VIA LASSO -> SVM)
# ==========================================
# Use Lasso to filter features first, then pass selected features into SVM
selected_features <- rownames(lasso_coefs)[lasso_coefs[,1] != 0]
selected_features <- setdiff(selected_features, "(Intercept)")

if(length(selected_features) > 0) {
  X_train_reduced <- X_train[, selected_features, drop = FALSE]
  X_test_reduced  <- X_test[, selected_features, drop = FALSE]
  
  svm_lasso_model <- svm(x = X_train_reduced, y = y_train_factor, kernel = "linear")
  svm_lasso_preds <- predict(svm_lasso_model, newdata = X_test_reduced)
  
  cat("\n--- HYBRID LASSO + SVM EVALUATION ---\n")
  print(confusionMatrix(svm_lasso_preds, y_test_factor))
}

# ==========================================
# Direct Sentiment Score per Text
# ==========================================
# Calculates numerical polarity scores (positive - negative) for each row
#df$sentiment_score <- get_sentiment(df$txt, method = "syuzhet")

# Optional: Categorize scores into Positive, Negative, Neutral
#df$sentiment_label <- ifelse(df$sentiment_score > 0, "Positive",
#                            ifelse(df$sentiment_score < 0, "Negative", "Neutral"))

#print("Sample Sentiment Scores:")
#print(head(df[, c("date", "Is Price UP", "sentiment_score", "sentiment_label")]))
