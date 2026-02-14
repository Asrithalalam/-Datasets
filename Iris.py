#Import required libraries 
import pandas as pd              
import matplotlib.pyplot as plt  
import seaborn as sns            
import os                        

# 1. Load Dataset
df = pd.read_csv("iris.csv")      
print("Dataset Loaded Successfully")


df = df.drop(columns=['Id'])        #it is only an index column

print(df.head())
print("\nColumns:", df.columns)

# 2. Create visuals folder(for visualizations)
if not os.path.exists("visuals"):        
    os.makedirs("visuals")               
    
# 3. Correlation Heatmap (All Numeric Columns)

plt.figure(figsize=(8, 6))                                 
correlation = df.select_dtypes(include='number').corr()   #selects only numeric columns & calculates correlation between them.

sns.heatmap(correlation, annot=True)               #shows actual correlation values inside cells.   
plt.title("Correlation Heatmap - Iris Dataset")
plt.tight_layout()      
plt.savefig("visuals/correlation_heatmap.png")    #saving images in "visuals" folder   
plt.close()    
print("Saved: correlation_heatmap.png")


# 4. Pairplot (All Numeric Relationships by Species) .

sns.pairplot(df, hue="Species")            #colors points by species
plt.savefig("visuals/pairplot_species.png")
plt.close()
print("Saved: pairplot_species.png")


# 5. Scatter Plot: Petal Length vs Petal Width

plt.figure()                           
sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species")
plt.title("Petal Length vs Petal Width")
plt.tight_layout()
plt.savefig("visuals/petal_scatter.png")
plt.close()
print("Saved: petal_scatter.png")

# 6. Boxplot: Sepal Length by Species (Numerical vs Categorical)

plt.figure()
sns.boxplot(data=df, x="Species", y="SepalLengthCm")
plt.title("Sepal Length Distribution by Species")
plt.tight_layout()
plt.savefig("visuals/sepal_length_boxplot.png")
plt.close()
print("Saved: sepal_length_boxplot.png")

# 7. Average Measurements per Species (Bar Plot) - Numerical vs Categorical (mean comparison)

species_mean = df.groupby("Species").mean()    
species_mean.plot(kind="bar")                 
plt.title("Average Feature Values per Species")
plt.ylabel("Average Measurement")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("visuals/average_features_by_species.png")
plt.close()
print("Saved: average_features_by_species.png")

print("\nAll visualizations saved successfully!")
