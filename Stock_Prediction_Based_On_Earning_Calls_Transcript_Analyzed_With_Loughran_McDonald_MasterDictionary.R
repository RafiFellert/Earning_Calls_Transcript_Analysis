library(tidyverse)
library(tidytext)
library(caret)
library(randomForest)
library(pROC)
library(rsample)

#################################################
# 1. Load Loughran-McDonald Dictionary
#################################################

root_folder <- "C:/Users/felle/Documents/Rafi/text_mining/text_mining_project"

# Path to the LM Master Dictionary CSV
lm_file_path <- file.path(root_folder, "Loughran-McDonald_MasterDictionary_1993-2025.csv")
transcripts_file_path <- file.path(root_folder, "transcripts_with_returns.csv")

lm_dict_raw <- read_csv(lm_file_path)

# Extract Positive and Negative word sets
lm_dict <- lm_dict_raw %>%
  mutate(Word = tolower(Word)) %>%
  filter(Positive > 0 | Negative > 0) %>%
  mutate(sentiment = case_when(
    Positive > 0 ~ "positive",
    Negative > 0 ~ "negative"
  )) %>%
  select(word = Word, sentiment)

# Load Earnings Call Transcripts
transcripts_df <- read_csv(transcripts_file_path)

# ==============================================================================
# 3. Calculate Tone Features (Tidytext Approach)
# ==============================================================================

# Tokenize text into words
tokens <- transcripts_df %>%
  select(Date, Ticker, TXT) %>%
  unnest_tokens(word, TXT)

# Total word count per transcript
word_counts <- tokens %>%
  count(Date, Ticker, name = "total_words")

# Count Positive and Negative words using LM Dictionary
sentiment_counts <- tokens %>%
  inner_join(lm_dict, by = "word") %>%
  count(Date, Ticker, sentiment) %>%
  complete(Date, Ticker, sentiment = c("positive", "negative"), fill = list(n = 0)) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

# Compute sentiment ratios & metrics
tone_features <- word_counts %>%
  left_join(sentiment_counts, by = c("Date", "Ticker")) %>%
  replace_na(list(positive = 0, negative = 0)) %>%
  mutate(
    pos_ratio   = positive / total_words,
    neg_ratio   = negative / total_words,
    net_tone    = (positive - negative) / (positive + negative + 1e-5),
    lm_polarity = (positive - negative) / total_words
  )

# Merge tone metrics back to main dataset & standardise target variable
model_data <- transcripts_df %>%
  left_join(tone_features, by = c("Date", "Ticker")) %>%
  mutate(
    # Ensure binary target is 0/1 numeric and a factor for evaluation
    Target_Numeric = ifelse(`Is Price UP` %in% c("YES", "1", 1), 1, 0),
    Target_Factor  = factor(Target_Numeric, levels = c(0, 1), labels = c("NO", "YES"))
  ) %>%
  select(Date, Ticker, Target_Numeric, Target_Factor, total_words, pos_ratio, neg_ratio, net_tone, lm_polarity) %>%
  drop_na(Target_Numeric)

# ==============================================================================
# 4. Train/Test Split
# ==============================================================================

set.seed(42)

data_split <- initial_split(model_data, prop = 0.80, strata = Target_Factor)
train_data <- training(data_split)
test_data  <- testing(data_split)

message(sprintf("Training rows: %d | Testing rows: %d", nrow(train_data), nrow(test_data)))

# ==============================================================================
# 5. Model Training (Logistic Regression)
# ==============================================================================

# Note: pos_ratio and total_words are used to prevent multicollinearity with net_tone/lm_polarity
log_model <- glm(
  Target_Numeric ~ pos_ratio + neg_ratio + total_words, 
  data = train_data,
  family = "binomial"
)

summary(log_model)

# ==============================================================================
# 6. Prediction & Classification Evaluation
# ==============================================================================

# Predict probabilities on unseen test data
test_predictions <- test_data %>%
  mutate(
    pred_prob  = predict(log_model, newdata = test_data, type = "response"),
    pred_class = factor(ifelse(pred_prob >= 0.5, "YES", "NO"), levels = c("NO", "YES"))
  )

# Confusion Matrix & Classification Metrics (Accuracy, Sensitivity, Specificity)
conf_matrix <- confusionMatrix(test_predictions$pred_class, test_predictions$Target_Factor)
print(conf_matrix)

# Calculate ROC-AUC Score
roc_obj <- roc(test_predictions$Target_Numeric, test_predictions$pred_prob)
message(sprintf("Test AUC Score: %.4f", auc(roc_obj)))