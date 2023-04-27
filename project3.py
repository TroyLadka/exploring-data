#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:46:19 2023

@author: troyladka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, pearsonr

questions = pd.read_csv('codereview-questions.csv', thousands=',', parse_dates=['Question_Post_Time'])
answers = pd.read_csv('codereview-answers.csv', thousands=',')

merge = pd.merge(answers, questions, on='Question_ID', how='outer')

bools = merge.select_dtypes(include='bool').columns
cols = merge.select_dtypes(include=np.number).columns

moments = []
for col in bools.union(cols):
    mean = merge[col].mean()
    std = merge[col].std()
    sk = skew(merge[col].fillna(0).astype(int))
    kurt = kurtosis(merge[col].fillna(0).astype(int))
    moments.append([col, mean, std, sk, kurt])
   
moments_df = pd.DataFrame(moments, columns=['Variable_Name', 'Mean', 'Standard_Deviation', 'Skew', 'Kurtosis'])
moments_df.to_csv('moments.csv', index=False)

question_author_counts = merge.groupby('Author_ID_y').size()
top_question_authors = question_author_counts.sort_values(ascending=False).head(10)

question_authors_rep = merge[['Author_ID_y', 'Author_Rep_y']].drop_duplicates(subset='Author_ID_y')

top_question_authors_rep = question_authors_rep.set_index('Author_ID_y').loc[top_question_authors.index]
top_question_authors_rep = top_question_authors_rep[top_question_authors_rep.index != 0]
top_question_authors_rep.index.name = 'Question_Author_ID'
top_question_authors_rep.columns = ['Reputation']

top_question_authors_rep.to_csv('top_question_authors_with_reputation.csv')

def tags_procedure(filtered_merge, filter_name):
    tags = filtered_merge[['Tag_1', 'Tag_2', 'Tag_3', 'Tag_4', 'Tag_5']].melt(var_name='Tag_Position', value_name='Tag_Name')
    tag_counts = tags.groupby('Tag_Name').size().sort_values(ascending=False).head(10)
    if tag_counts.drop(' ').any():
        tag_counts = tag_counts.drop(' ')
    tag_counts.to_csv(f'{filter_name}_top_tags.csv', header=['Tag_Count'])
   
tags_procedure(merge, 'unflitered')

closed = merge[merge['Question_Closed']==True]
not_closed = merge[merge['Question_Closed']==False]
edited = merge[merge['Edited']==True]
not_edited = merge[merge['Edited']==False]
accepted = merge[merge['Answer_Accepted_y']==True]
not_accepted = merge[merge['Answer_Accepted_y']==False]

tags_procedure(closed, 'closed_questions')
tags_procedure(not_closed, 'not_closed_questions')
tags_procedure(edited, 'edited_questions')
tags_procedure(not_edited, 'not_edited_questions')
tags_procedure(accepted, 'accepted_questions')
tags_procedure(not_accepted, 'not_accepted_questions')

questions.set_index('Question_Post_Time', inplace=True)

daily_posts = questions.resample('D').count()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(daily_posts.index, daily_posts['Question_ID'])
ax.set_xlabel('Date')
ax.set_ylabel('Number of Posts')
ax.set_title('Daily Posts')

fig.savefig('daily_posts.png')
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

ax1.hist(merge['Question_Score'], alpha=0.5, bins=20, color='red', label='Question Scores')
ax1.set_xlabel('Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Question Scores and Answer Scores')

ax1.hist(merge['Answer_Score'], alpha=0.5, bins=20, color='blue', label='Answer Scores')
ax1.legend(loc='best')

ax2.hist(merge['Number_Of_Comments_y'], alpha=0.5, bins=20, color='black')
ax2.set_xlabel('Number of Comments (Questions)')
ax2.set_ylabel('Frequency')
ax2.set_title('Number of Comments on Questions')

ax3.hist(merge['Number_Of_Comments_x'], alpha=0.5, bins=20, color='green')
ax3.set_xlabel('Number of Comments (Answers)')
ax3.set_ylabel('Frequency')
ax3.set_title('Number of Comments on Answers')

plt.savefig('histogram.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merge['Question_Score'], merge['Answer_Score'], alpha=0.5)
ax.set_xlabel('Question Score')
ax.set_ylabel('Answer Score')
ax.set_title('Question Scores vs. Answer Scores')

new_merge = merge.dropna(subset=['Question_Score', 'Answer_Score'])

corr, pval = pearsonr(new_merge['Question_Score'].dropna(), new_merge['Answer_Score'].dropna())

ax.text(0.05, 0.95, f'Correlation: {corr}\nP-value: {pval}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

plt.savefig('question_scores_vs_answer_scores.png')
plt.show()


merge_numeric = merge.select_dtypes(include=np.number)
merge_numeric = merge_numeric.drop(['Question_ID', 'Author_ID_x', 'Author_ID_y'], axis=1)
correlation_matrix = merge_numeric.corr()
correlation_matrix = correlation_matrix.mask(pd.Series(False, index=correlation_matrix.index), other=np.nan)
correlation_matrix = merge_numeric.corr().where(np.triu(np.ones(merge_numeric.corr().shape), k=1).astype(bool))
np.fill_diagonal(correlation_matrix.values, np.nan)
positive_correlation = correlation_matrix.stack().nlargest(2)
negative_correlation = correlation_matrix.stack().nsmallest(1)
print("Two most positively correlated variables:")
print(positive_correlation)
print("\nTwo most negatively correlated variables:")
print(negative_correlation)
