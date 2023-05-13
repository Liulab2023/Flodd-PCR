import os 
import glob 
import pandas as pd 
import skfuzzy as fuzz 
import numpy as np 
import math 
import matplotlib.pyplot as plt 


# Traverse the folder and get all file names that meet the conditions 
csv_files = []
for root, dirs, files in os.walk("D:/SD3 HPV16"):
    for file in files:
        if file.endswith(".csv") and "_cleaned" not in file:
            csv_files.append(os.path.join(root, file))


# Threshold setting
x = 15000 


# Process each file that meets the conditions
for file in csv_files:
    # Read CSV file
    try:
        df = pd.read_csv(file, usecols=["FAM", "CY5"])
    except ValueError:
        print("Error: file {} has no header.".format(file))
        continue 
    data = df.iloc[:, 1].values.reshape(-1, 1)
    num_data_before = len(data)


    # Perform fuzzy clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, 2, m=2, error=0.005, maxiter=1000, init=None)


    # Calculate the fuzziness of each data point
    fuzzy_membership = u.max(axis=0)


    # Calculate the average of the second column of class 0 and class 1
    mean_class_0 = df.iloc[np.where(np.argmax(u, axis=0) == 0)[0], 1].mean()
    mean_class_1 = df.iloc[np.where(np.argmax(u, axis=0) == 1)[0], 1].mean()


    # If the average of the second column of class 0 is greater than that of class 1, swap the labels of class 0 and class 1
    if mean_class_0 > mean_class_1:
        u = 1 - u 


    # Set the threshold and remove data points with fuzziness less than the threshold
    threshold = 0.65 
    keep_indices = np.where(fuzzy_membership >= threshold)[0]
    data = data[keep_indices]
    df = df.iloc[keep_indices, :]
    u = u[:, keep_indices]
    fuzzy_membership = fuzzy_membership[keep_indices]


    # Save the cleaned data to csv
    cleaned_csv_file = os.path.splitext(file)[0] + "_cleaned.csv" 
    df.to_csv(cleaned_csv_file, index=False)


    num_data_gt_16000 = len(df[df.iloc[:, 0] > x])
    ratio_gt_16000 = num_data_gt_16000 / len(data)


    # Rename the output result file according to the executed file name
    output_file = os.path.splitext(file)[0] + "_result.txt" 


    # Draw a 2D scatter plot
    plt.scatter(df.iloc[:,1], df.iloc[:,0], c=np.argmax(u,axis=0))


    # Set horizontal and vertical axis labels
    plt.xlabel('CY5')
    plt.ylabel('FAM')


    # Set the title of the figure
    plt.title('Clustering Result')


    # Save the image file
    output_image = os.path.splitext(file)[0] + "_result.png" 
    plt.savefig(output_image)
    plt.close()


    # Calculate the positive rate after modification, and calculate the positive rate for each of the two classes
    ratio_class_1 = len(df[(df.iloc[:,0]>x) & (np.argmax(u,axis=0)==0)]) / len(df[np.argmax(u,axis=0)==0])
    ratio_class_2     = len(df[(df.iloc[:,0]>x) & (np.argmax(u,axis=0)==1)]) / len(df[np.argmax(u,axis=0)==1])

    # Calculate the average copy number
    if ratio_class_1 == 1 or ratio_class_2 == 1:
        if ratio_class_1 == 1:
            avg_copy_number_class_1 = '-' 
            avg_copy_number_class_2 = -1 * math.log(1-ratio_class_2)
        else:
            avg_copy_number_class_1 = -1 * math.log(1-ratio_class_1)
            avg_copy_number_class_2 = '-' 
    else:
        avg_copy_number_class_1 = -1 * math.log(1-ratio_class_1)
        avg_copy_number_class_2 = -1 * math.log(1-ratio_class_2)

    # Output the results to a new text file
    with open(output_file, "w") as f:
        f.write("Number of data before cleaning: {}\n".format(num_data_before))
        f.write("Number of data after cleaning: {}\n".format(len(data)))
        f.write("Proportion of data retained: {}\n".format(len(data) / num_data_before))
        f.write("Threshold for positivity: {}\n".format(x))
        f.write("Class 1 data volume: {}\n".format(len(df[np.argmax(u,axis=0)==0])))
        f.write("Class 2 data volume: {}\n".format(len(df[np.argmax(u,axis=0)==1])))
        f.write("Class 1 positivity rate: {}\n".format(ratio_class_1))
        f.write("Class 2 positivity rate: {}\n".format(ratio_class_2))
        f.write("Average copy number of class 1: {}\n".format(avg_copy_number_class_1))
        f.write("Average copy number of class 2: {}\n".format(avg_copy_number_class_2))
        f.write("Results from folder: {}\n".format(os.path.dirname(file)))
        f.write("Results from file: {}\n\n".format(os.path.basename(file)))

