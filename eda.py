import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
print("Loading dataset for EDA...")
df = pd.read_csv("PS2_Dataset.csv")
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")
print(f"Dataset info:")
print(df.info())

# Create EDA directory
import os
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# ================================
# 1. BASIC DATA OVERVIEW
# ================================

print("\n" + "="*50)
print("1. BASIC DATA OVERVIEW")
print("="*50)

# Missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Data types
print("\nData Types:")
print(df.dtypes.value_counts())

# Basic statistics for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumerical Columns: {list(numerical_cols)}")
print("\nBasic Statistics:")
print(df[numerical_cols].describe())

# ================================
# 2. TARGET VARIABLE ANALYSIS
# ================================

print("\n" + "="*50)
print("2. TARGET VARIABLE ANALYSIS")
print("="*50)

# Career distribution
career_counts = df['Suggested Job Role'].value_counts()
print("\nCareer Distribution:")
print(career_counts)

# Plot career distribution
plt.figure(figsize=(12, 8))
career_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Career Roles', fontsize=16, fontweight='bold')
plt.xlabel('Career Roles', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/career_distribution.png', dpi=300, bbox_inches='tight')
print("Career distribution plot saved")
plt.show()

# Career distribution pie chart
plt.figure(figsize=(10, 10))
plt.pie(career_counts.values, labels=career_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Career Distribution - Pie Chart', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/career_pie_chart.png', dpi=300, bbox_inches='tight')
print("Career pie chart saved")
plt.show()

# ================================
# 3. NUMERICAL FEATURES ANALYSIS
# ================================

print("\n" + "="*50)
print("3. NUMERICAL FEATURES ANALYSIS")
print("="*50)

# Distribution of numerical features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')

for i, col in enumerate(numerical_cols[:4]):
    row = i // 2
    col_idx = i % 2
    
    axes[row, col_idx].hist(df[col], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[row, col_idx].set_title(f'Distribution of {col}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
print("Numerical distributions plot saved")
plt.show()

# Box plots for numerical features by career
for col in numerical_cols:
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='Suggested Job Role', y=col, palette='Set2')
    plt.title(f'{col} by Career Role', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'eda_plots/{col}_by_career_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"{col} boxplot saved")
    plt.show()

# ================================
# 4. CORRELATION ANALYSIS
# ================================

print("\n" + "="*50)
print("4. CORRELATION ANALYSIS")
print("="*50)

# Create a copy for correlation analysis
df_corr = df.copy()

# Encode categorical variables for correlation
from sklearn.preprocessing import LabelEncoder
categorical_cols = df_corr.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

# Correlation matrix
correlation_matrix = df_corr.corr()

# Plot correlation heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved")
plt.show()

# Top correlations with target variable
target_corr = correlation_matrix['Suggested Job Role'].abs().sort_values(ascending=False)
print("\nTop features correlated with Career Role:")
print(target_corr.head(10))
# ================================
# 5. CATEGORICAL FEATURES ANALYSIS
# ================================

print("\n" + "="*50)
print("5. CATEGORICAL FEATURES ANALYSIS")
print("="*50)

categorical_features = ['self-learning capability?', 'Extra-courses did', 'certifications', 
                       'workshops', 'Management or Technical', 'interested career area']

for feature in categorical_features:
    if feature in df.columns:
        print(f"\n{feature} distribution:")
        print(df[feature].value_counts())
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df[feature], df['Suggested Job Role'])
        
        # Plot stacked bar chart
        plt.figure(figsize=(14, 8))
        crosstab.plot(kind='bar', stacked=True, colormap='tab10')
        plt.title(f'{feature} vs Career Role', fontsize=14, fontweight='bold')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Career Role', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 🔧 FIX: Clean filename by removing invalid characters
        clean_filename = feature.replace('?', '').replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(f'eda_plots/{clean_filename}_vs_career.png', dpi=300, bbox_inches='tight')
        print(f" {feature} vs career plot saved")
        plt.show()

# ================================
# 6. SKILLS ANALYSIS
# ================================

print("\n" + "="*50)
print("6. SKILLS ANALYSIS")
print("="*50)

# Skills comparison by career
skills_cols = ['Logical quotient rating', 'coding skills rating', 'public speaking points']

# Average skills by career
skills_by_career = df.groupby('Suggested Job Role')[skills_cols].mean()
print("\nAverage Skills by Career:")
print(skills_by_career.round(2))

# Plot skills radar chart for top careers
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(career_counts.head(4).index),
    specs=[[{"type": "polar"}, {"type": "polar"}],
           [{"type": "polar"}, {"type": "polar"}]]
)

colors = ['red', 'blue', 'green', 'orange']
top_careers = career_counts.head(4).index

for i, career in enumerate(top_careers):
    career_data = df[df['Suggested Job Role'] == career][skills_cols].mean()
    
    fig.add_trace(go.Scatterpolar(
        r=list(career_data.values) + [career_data.values[0]],  # Close the polygon
        theta=list(career_data.index) + [career_data.index[0]],
        fill='toself',
        name=career,
        line_color=colors[i]
    ), row=(i//2)+1, col=(i%2)+1)

fig.update_layout(height=800, title_text="Skills Profile by Career", title_x=0.5)
fig.write_html('eda_plots/skills_radar_chart.html')
fig.show()
print("Skills radar chart saved")

# ================================
# 7. ADVANCED INSIGHTS
# ================================

print("\n" + "="*50)
print("7. ADVANCED INSIGHTS")
print("="*50)

# Hackathons impact analysis
plt.figure(figsize=(12, 6))
hackathon_career = df.groupby(['Suggested Job Role'])['hackathons'].mean().sort_values(ascending=False)
hackathon_career.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Average Hackathons Attended by Career', fontsize=14, fontweight='bold')
plt.xlabel('Career Role')
plt.ylabel('Average Hackathons')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots/hackathons_by_career.png', dpi=300, bbox_inches='tight')
print("Hackathons analysis saved")
plt.show()

# Skills distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Skills Distribution Across All Students', fontsize=16, fontweight='bold')

for i, skill in enumerate(skills_cols):
    axes[i].hist(df[skill], bins=10, color='lightblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{skill}')
    axes[i].set_xlabel('Rating')
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(df[skill].mean(), color='red', linestyle='--', label=f'Mean: {df[skill].mean():.1f}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('eda_plots/skills_distribution_overview.png', dpi=300, bbox_inches='tight')
print("Skills distribution overview saved")
plt.show()

# ================================
# 8. KEY INSIGHTS SUMMARY
# ================================

print("\n" + "="*50)
print("8. KEY INSIGHTS SUMMARY")
print("="*50)

print("\nKEY FINDINGS:")
print("="*30)

print(f"1. Dataset has {df.shape[0]} students with {df.shape[1]} features")
print(f"2. Most common career: {career_counts.index[0]} ({career_counts.iloc[0]} students)")
print(f"3. Least common career: {career_counts.index[-1]} ({career_counts.iloc[-1]} students)")
print(f"4. Average coding skills: {df['coding skills rating'].mean():.1f}/10")
print(f"5. Average logical quotient: {df['Logical quotient rating'].mean():.1f}/10")
print(f"6. Average public speaking: {df['public speaking points'].mean():.1f}/10")

# Class imbalance ratio
imbalance_ratio = career_counts.max() / career_counts.min()
print(f"7. Class imbalance ratio: {imbalance_ratio:.1f}:1 (explains why we needed SMOTE)")

# Most correlated features with career
top_features = target_corr.head(6).index[1:]  # Exclude target itself
print(f"8. Most predictive features: {', '.join(top_features)}")

print(f"\nAll EDA plots saved in 'eda_plots/' directory")
print("EDA Analysis Complete!")

# Save insights to text file
with open('eda_plots/eda_insights.txt', 'w') as f:
    f.write("EDA INSIGHTS SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Career Distribution:\n{career_counts}\n\n")
    f.write(f"Skills Averages:\n{df[skills_cols].mean()}\n\n")
    f.write(f"Top Correlated Features:\n{target_corr.head(10)}\n")

print("EDA insights saved to 'eda_plots/eda_insights.txt'")