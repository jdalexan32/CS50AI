import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    
    # Define a mapping for month abbreviations to their corresponding index
    month_mapping = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        
        for row in reader:
            # Extract values from the row
            administrative, admin_duration, informational, info_duration, product_related, product_duration, \
            bounce_rates, exit_rates, page_values, special_day, month, os, browser, region, traffic_type, \
            visitor_type, weekend, revenue = row
            
            # Convert values to appropriate types
            evidence_row = [
                int(administrative), float(admin_duration), int(informational), float(info_duration),
                int(product_related), float(product_duration), float(bounce_rates), float(exit_rates),
                float(page_values), float(special_day), month_mapping.get(month[:3], 0), int(os), int(browser),
                int(region), int(traffic_type), 1 if visitor_type == 'Returning_Visitor' else 0,
                1 if weekend == 'TRUE' else 0  # Convert 'TRUE' to 1 and 'FALSE' to 0
            ]
            
            label = 1 if revenue == 'TRUE' else 0
            
            # Append evidence and label to their respective lists
            evidence.append(evidence_row)
            labels.append(label)    
    
    """
    # Print the first evidence list
    print(f"First evidence list: {evidence[0]}")
    # Print the first label
    print(f"First lable: {labels[0]}")
    
    """
    
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Create DataFrames from the labels and predictions lists
    df = pd.DataFrame({'labels': labels, 'predictions': predictions})

    # Calculate True Positives, True Negatives, False Positives, and False Negatives
    true_positives = df[(df['labels'] == 1) & (df['predictions'] == 1)].shape[0]
    true_negatives = df[(df['labels'] == 0) & (df['predictions'] == 0)].shape[0]
    false_positives = df[(df['labels'] == 0) & (df['predictions'] == 1)].shape[0]
    false_negatives = df[(df['labels'] == 1) & (df['predictions'] == 0)].shape[0]

    """
    # Print the number of purchases (True positives) and non-purchases (True negatives)
    print("Number of purchases (True positives):", true_positives)
    print("Number of non-purchases (True negatives):", true_negatives)

    """

    # Calculate sensitivity and specificity
    if (true_positives + false_negatives) == 0:
        sensitivity = 0.0
    else:
        sensitivity = true_positives / (true_positives + false_negatives)

    if (true_negatives + false_positives) == 0:
        specificity = 0.0
    else:
        specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
