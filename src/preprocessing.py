import gc
import re
import pandas as pd
from hazm import Normalizer, InformalNormalizer, Lemmatizer, stopwords_list


def get_stop_words():
    # Load Persian stopwords and preserve descriptive words that may carry sentiment information
    stop_words = stopwords_list()
    descriptive_words = {
        'بزرگ', 'بسیار', 'بسیاری', 'بهتر', 'بهترین', 'بیش', 'بیشتر', 'بیشتری',
        'حل', 'خاص', 'خوب', 'خوبی', 'خیلی', 'زیاد', 'زیادی', 'شروع', 'عالی',
        'لازم', 'متفاوت', 'متاسفانه', 'مشخص', 'مناسب', 'مهم', 'نظیر', 'نیاز',
        'پیدا', 'کافی', 'کامل', 'کاملا', 'کم', 'کمی', 'کند', 'اثر', 'بالا',
        'تغییر', 'خطر', 'رشد', 'کوچک', 'آخرین', 'ابتدا', 'اجرا', 'جدی',
        'جدید', 'نظر', 'کلی', 'اغلب', 'عالیه', 'نباید', 'نبود', 'ندارد',
        'نیست', 'نمی‌شود'
    }
    return set(stop_words) - descriptive_words


def clean_and_process(text, normalizer, informal_normalizer, lemmatizer, stop_words):
    """
    Complete Persian text preprocessing pipeline:
    Character cleaning -> Normalization -> Colloquial-to-formal conversion -> Lemmatization -> Stop word removal
    """
    # Remove non-Persian characters and normalize whitespace
    text = re.sub(r'[^آ-ی\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalize Persian text
    text = normalizer.normalize(text)

    # Normalize informal words and generate possible normalized forms
    parsed = informal_normalizer.normalize(text)  # [[[opt1, opt2, ...], ...]]

    final_words = []

    for sentence in parsed:
        for word_options in sentence:
            # Select the most appropriate form from the generated alternatives
            if len(word_options) > 2:
                chosen = None

                for w in word_options:
                    if w.endswith(' است') and re.match(r'^(.*[^ه])[\s\u200c]است$', w):
                        chosen = w
                        break

                if chosen is None:
                    chosen = word_options[1]
            else:
                chosen = word_options[0]

            for token in chosen.split():
                if not token.strip():
                    continue

                # Preserve negation while applying lemmatization
                if re.match(r'^ن[^ن]', token):
                    lemma = 'ن' + lemmatizer.lemmatize(token[1:])
                else:
                    lemma = lemmatizer.lemmatize(token)

                # Remove additional information returned by the lemmatizer
                lemma_clean = lemma.split('#')[0] if '#' in lemma else lemma

                # Remove stopwords while keeping sentiment-related words
                if lemma_clean not in stop_words:
                    final_words.append(lemma_clean)

    return ' '.join(final_words)


def main():
    # Define input and output file paths
    pd.set_option('display.max_colwidth', None)

    input_path = "../data/digikala-comments.csv"
    output_path = "../data/digikala_preprocessed.csv"

    # Load only the columns required for text preprocessing and classification
    target_df = pd.read_csv(
        input_path,
        usecols=['title', 'body', 'advantages', 'disadvantages', 'recommendation_status']
    )

    # Combine all available text fields into a single review text
    text_columns = ["title", "body", "advantages", "disadvantages"]
    target_df["body"] = (
        target_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    )

    # Keep only the review text and target label
    target_df = target_df[["body", "recommendation_status"]]

    # Remove empty reviews and samples with missing target labels
    target_df = target_df[target_df["body"].str.strip().ne("")]
    target_df = target_df.dropna(subset=["recommendation_status"])

    # Remove duplicate reviews based on the combined review text
    target_df = target_df.drop_duplicates(subset=["body"])

    # Separate the majority class from the other classes
    recommended_df = target_df[target_df["recommendation_status"] == "recommended"]
    other_df = target_df[target_df["recommendation_status"] != "recommended"]

    # Downsample the majority class to reduce class imbalance
    n_sample = min(1_000_000, len(recommended_df))
    recommended_sample = recommended_df.sample(n=n_sample, random_state=42)

    target_df = pd.concat([recommended_sample, other_df], ignore_index=True)

    # Release unused DataFrames to reduce memory usage
    del recommended_df, other_df, recommended_sample
    gc.collect()

    # Convert string labels to numerical class IDs
    label_mapping = {
        "not_recommended": 0,
        "no_idea": 1,
        "recommended": 2
    }
    target_df["recommendation_status"] = target_df["recommendation_status"].map(label_mapping)

    # Initialize Persian text preprocessing tools
    normalizer = Normalizer()
    informal_normalizer = InformalNormalizer()
    lemmatizer = Lemmatizer()
    stop_words = get_stop_words()

    # Apply the complete preprocessing pipeline to all reviews
    comments = target_df["body"].apply(
        lambda x: clean_and_process(
            x,
            normalizer,
            informal_normalizer,
            lemmatizer,
            stop_words
        )
    )

    # Create the final dataset containing processed text and numerical labels
    data = pd.DataFrame({
        'comments': comments,
        'label': target_df['recommendation_status']
    })

    # Save the preprocessed dataset
    data.to_csv(output_path, index=False, encoding="utf-8")

    print(f"File saved to {output_path}")


if __name__ == "__main__":
    main()
