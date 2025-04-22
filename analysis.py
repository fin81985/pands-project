# Project 2025: Iris Dataset Analysis

# Module: Programming and Scripting
# Author: Finian Doonan


# Data manipulation
import numpy as np # For numerical operations
import pandas as pd # For data manipulation and analysis

# Machine Learning Library
import sklearn # For machine learning algorithms
from sklearn import datasets  # For loading datasets

# Plotting
import matplotlib.pyplot as plt # For basic plotting
import seaborn as sns  # Visualization library based on matplotlib
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

# Load the iris dataset.
df = pd.read_csv("iris_dataset/iris.data")

# Have a look.
df

# Describe the data set.
df.describe()

# look at the data.
df

# Look at the keys
df.keys()

# Rename columns to better names
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# The species of iris
df['species']

# Shape
df['species'].shape

# The sepal length
df['sepal_length']

# The petal lenght
df['petal_length']

# the petal width
df['petal_width']

# Statistics
stats = df.describe().T[['mean', 'min', 'max', 'std']]
stats['median'] = df.median(numeric_only=True) # select only the numeric columns

# Print results
print(stats)

#  summary statistics
stats = {
    "sepal_length": np.random.normal(5.843333, 0.828066, 150),
    "sepal_width": np.random.normal(3.054000, 0.433594, 150),
    "petal_length": np.random.normal(3.758667, 1.764420, 150),
    "petal_width": np.random.normal(1.198667, 0.763161, 150)
}

# Plot histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

# create histogram for each feature
for idx, (feature, values) in enumerate(stats.items()):
    axes[idx].hist(values, bins=20, edgecolor='black', alpha=0.7)
    axes[idx].set_title(feature)
    axes[idx].set_xlabel("Value (cm)")
    axes[idx].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Generate synthetic class labels with names
class_names = ["Setosa", "Versicolor", "Virginica"]
classes = np.random.choice(class_names, size=150)
colors = {'Setosa': 'r', 'Versicolor': 'g', 'Virginica': 'b'}

# Scatter plot of sepal_length vs petal_length
plt.figure(figsize=(8, 6))
for class_label in np.unique(classes):
    plt.scatter(
        stats["sepal_length"][classes == class_label],
        stats["petal_length"][classes == class_label],
        color=colors[class_label],
        label=f'Class {class_label}',
        alpha=0.7
    )

# Add labels and legend
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Scatter Plot of Sepal Length vs Petal Length")
plt.legend()
plt.show()

# Scatter plot of sepal_length vs petal_length
plt.figure(figsize=(8, 6))
for class_label in np.unique(classes):
    plt.scatter(
        stats["sepal_length"][classes == class_label],
        stats["petal_length"][classes == class_label],
        color=colors[class_label],
        label=f'Class {class_label}',
        alpha=0.7
    )
# Fit a regression line using numpy.polyfit
x = stats["sepal_length"]
y = stats["petal_length"]

# Fit a regression line
coefficients = np.polyfit(x, y, 1)

# Create a polynomial object
polynomial = np.poly1d(coefficients)
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)

# Plot the regression line
plt.plot(x_line, y_line, color='k', linestyle='-', linewidth=3, label='Regression Line')

# Add labels and title
plt.xlabel("Sepal Length (cm) ")
plt.ylabel("Petal Length (cm)")
plt.title("Scatter Plot of Sepal Length vs Petal Length")
plt.legend()
plt.show()

# Create box-plots for petal lengths of each class
plt.figure(figsize=(8, 6)) 

# Extract petal lengths for each class
petal_lengths_by_class = [stats["petal_length"][classes == class_label] for class_label in np.unique(classes)]

# Create box-plots
plt.boxplot(petal_lengths_by_class, tick_labels=[f'Class {c}' for c in np.unique(classes)])


# Add labels
plt.xlabel("Class")
plt.ylabel("Petal Length (cm)")
plt.title("Box Plot of Petal Lengths for Each Class")
plt.show()

# Convert to DataFrame
df = pd.DataFrame(stats)

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add title
plt.title("Correlation Heatmap of Iris Features (cm)", fontsize=14, fontweight='bold')

# Show plot
plt.show()

# Create pair plot
# Add species column separately to ensure proper assignment
df["species"] = np.random.choice(["setosa", "versicolor", "virginica"], 150)

# Ensure 'species' column exists
if "species" not in df.columns:
    print("Warning: 'species' column is missing from DataFrame")

# Create pair plot
sns.pairplot(df, hue="species", diag_kind="kde", markers=["o", "s", "D"])
plt.show()

# Create violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='species', y='petal_length', data=df)# add x and y axis

# Add title and labels
plt.title('Distribution of Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

# Show plot
plt.show()