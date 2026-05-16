# Part 3 Dataset: Customer Support Text Classification Dataset

## File
`customer_support_text_classification.csv`

## Goal
Build an NLP model to classify customer messages by sentiment.

## Target Column
- `sentiment_label`: `positive`, `neutral`, or `negative`

## Useful Columns
- `customer_message`: input text for NLP model
- `channel`: source channel of the ticket
- `word_count`: derived text length indicator
- `urgent_flag`: optional binary column for additional analysis

## Suggested Student Tasks
- Clean text
- Tokenize messages
- Create Bag of Words, TF-IDF, or sequence-based vectors
- Build a baseline model
- Build or describe an RNN/LSTM model
- Reflect on attention and transformers
