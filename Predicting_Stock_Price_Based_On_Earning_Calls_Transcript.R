# ==========================================
# Algorithms: 
#   1. LASSO  
#   2. SVM
# Feature Sets:
#   1. Loughran-McDonald Dictionary Features
#   2. Unigrams
#   3. Bigrams
#   4. Topic Modeling Only
#   5. Unigrams + Topic Modeling
#   6. Sentiment Only
#   7. Sentiment + Unigrams + Topic Modeling
#   8. TF-IDF Weighted Unigrams
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
library(tidyverse)
library(tidytext)

if(!requireNamespace("textstem", quietly = TRUE)) {
  install.packages("textstem")
}
if(!requireNamespace("lexicon", quietly = TRUE)) {
  install.packages("lexicon")
}
if(!requireNamespace("sentimentr", quietly = TRUE)) {
  install.packages("sentimentr")
}

# 1. Load Data & LM Dictionary
# ------------------------------------------------
root_folder <- "C:/Users/felle/Documents/Rafi/text_mining/text_mining_project"
transcripts_file_path <- file.path(root_folder, "transcripts_with_returns.csv")

df <- read.csv(transcripts_file_path, stringsAsFactors = FALSE)
df <- subset(df, `Is.Price.UP` %in% c("YES", "NO"))

# 2. Load and prepare Loughran-McDonald Dictionary
# ------------------------------------------------
lm_file_path <- file.path(root_folder, "Loughran-McDonald_MasterDictionary_1993-2025.csv")
lm_dict_raw <- read.csv(lm_file_path)

lm_dict <- lm_dict_raw %>%
  mutate(Word = tolower(Word)) %>%
  filter(Positive > 0 | Negative > 0) %>%
  mutate(sentiment = case_when(
    Positive > 0 ~ "positive",
    Negative > 0 ~ "negative"
  )) %>%
  select(word = Word, sentiment)

# 3. Text Preprocessing Function
# ------------------------------------------------
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

# 4. Build Base DTMs & Safely Filter Empty Rows
# ------------------------------------------------
dtm_unigram <- DocumentTermMatrix(corpus)

BigramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "))
}

dtm_bigram <- DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer))

categorial_param <- factor(df$`Is.Price.UP`, levels = c("NO", "YES"))

# 5. Construct Feature Sets
# ------------------------------------------------
# Feature 1: Loughran-McDonald Dictionary Features
tokens <- df %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, TXT) %>%
  unnest_tokens(word, TXT)

word_counts <- tokens %>%
  count(doc_id, name = "total_words")

sentiment_counts <- tokens %>%
  inner_join(lm_dict, by = "word") %>%
  count(doc_id, sentiment) %>%
  complete(doc_id = 1:nrow(df), sentiment = c("positive", "negative"), fill = list(n = 0)) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

lm_features_df <- data.frame(doc_id = 1:nrow(df)) %>%
  left_join(word_counts, by = "doc_id") %>%
  left_join(sentiment_counts, by = "doc_id") %>%
  replace_na(list(total_words = 0, positive = 0, negative = 0)) %>%
  mutate(
    pos_ratio   = ifelse(total_words > 0, positive / total_words, 0),
    neg_ratio   = ifelse(total_words > 0, negative / total_words, 0),
    net_tone    = (positive - negative) / (positive + negative + 1e-5),
    lm_polarity = ifelse(total_words > 0, (positive - negative) / total_words, 0)
  )

X_lm_dictionary <- as.matrix(lm_features_df %>% select(pos_ratio, neg_ratio, net_tone, lm_polarity, total_words))

# Feature 2: Unigrams
dtm_uni_sparse <- removeSparseTerms(dtm_unigram, 0.98)
X_unigram      <- as.matrix(dtm_uni_sparse)

# Feature 3: Bigrams
dtm_bi_sparse <- removeSparseTerms(dtm_bigram, 0.98)
X_bigram      <- as.matrix(dtm_bi_sparse)

# Feature 4: LDA Topics (20 Topics)
lda_model  <- LDA(dtm_unigram, k = 10, method = "Gibbs", 
                  control = list(seed = 1, burnin = 500, iter = 1000))

X_topics_raw <- posterior(lda_model)$topics # Raw proportions (0.0 to 1.0)
colnames(X_topics_raw) <- paste0("Topic_", 1:10)

X_topics_scaled <- X_topics_raw

# Feature 5: Combined Unigrams + Raw Topic Proportions
X_unigram_topics <- cbind(X_unigram, X_topics_raw)

# Feature 6: Calculate sentiment
sent_res <- sentiment_by(df$TXT, by = 1:nrow(df))

X_sentiment <- as.matrix(data.frame(
  Sentiment_Score = sent_res$ave_sentiment,
  Word_Count      = sent_res$word_count
))

# Feature 7: # Combine Sentiment + Unigrams + Raw Topic Proportions
X_sentiment_unigram_topics <- cbind(X_unigram_topics, X_sentiment)

# Feature 8: TF-IDF (Built directly from row-filtered dtm_unigram)
dtm_tfidf        <- weightTfIdf(dtm_unigram)
dtm_tfidf_sparse <- removeSparseTerms(dtm_tfidf, 0.98)
X_tfidf          <- as.matrix(dtm_tfidf_sparse)

feature_sets <- list(
  "1. Loughran-McDonald Dictionary Features" = X_lm_dictionary,
  "2. Unigrams"                              = X_unigram,
  "3. Bigrams"                               = X_bigram,
  "4. Topics Only"                           = X_topics_scaled,
  "5. Unigrams + Topic Modeling"             = X_unigram_topics,
  "6. Sentiment Only"                        = X_sentiment,
  "7. Sentiment + Unigrams + Topic Modeling" = X_sentiment_unigram_topics,
  "8. TF-IDF Weighted Unigrams"            = X_tfidf
)

# 6. Model Execution Loop
# ------------------------------------------------
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
  svm <- svm(x = X_train, y = y_train, gamma = 1 / ncol(X_train))
  svm_pred <- predict(svm, X_test)
  cat("\n--- SVM ---\n")
  print(confusionMatrix(svm_pred, y_test))
}

for (name in names(feature_sets)) {
  run_classifiers(feature_sets[[name]], categorial_param, name)
}
