# Mental Health & Age Analysis Among U.S. Adults

### Data
**Behavioral Risk Factor Surveillance System (2018):** Also located on [Kaggle.com - Behavioral Risk Factor Surveillance System (2018)](https://www.kaggle.com/datasets/rosaaestrada/behavioral-risk-factor-surveillance-system)

------------------------------------------------------------------------------------------------------------------------

### Project Overview and Objectives
Analyze the relationship between mental health and age in adults between the ages of 18 and 99 in the United States.

🟣 **Research Question:** What ages experience the most "no good" mental health days?

🟣 **Alternative Hypothesis (H1):** There is no significant difference in the number of 'no good' mental health days reported across different age groups.

🟣 **Null Hypothesis (H0):** There is a significant difference in the number of 'no good' mental health days reported across different age groups.

------------------------------------------------------------------------------------------------------------------------

**Methodology:** 

This project employs a structured methodology consisting of (Data cleaning, Exploratory Data Analysis (EDA), and feature engineering). Following those steps, statistical analysis is conducted utilizing Pearson's Chi-Square, Analysis of Variance (ANOVA), and Correlation Matrix. 

------------------------------------------------------------------------------------------------------------------------

**Statistics:**
- 1 in 5 adults have a mental health condition; that is more than the population of New York and Florida combined, making up more than 40 million Americans (Mental Health America, 2023).
- In 2018, 19% of adults experienced a mental health illness (Terlizzi & Zablotsky, 2020).
- In 2019, 20% of adults were living with a mental health illness, and in 2020, the percentage increased to 40% (The Blackberry Center, 2020).
- The National Alliance on Mental Health Illness found that 50% of all lifetime mental illnesses begin at the age of 14, and 75% begin at the age of 24 (2023).

**References:**
- Mental Health America. (2023). The State of Mental Health in America 2018. Mental Health America. Retrieved March 23, 2023, from https://www.mhanational.org/issues/state-mental-health-america-2018
- National Alliance on Mental Illness. (n.d.). Mental Health Conditions. NAMI. Retrieved April 8, 2024, from https://www.nami.org/About-Mental-Illness/Mental-Health-Conditions
- Terlizzi, E. P., & Zablotsky, B. (2020, September). Mental Health Treatment Among Adults: United States, 2019. PubMed. Retrieved April 8, 2024, from https://pubmed.ncbi.nlm.nih.gov/33054921/
- The Blackberry Center. (2020, December 28). Mental Health Statistics in 2020 Compared to 2019 - BBC. The Blackberry Center. Retrieved March 27, 2023, from https://www.theblackberrycenter.com/mental-health-statistics-in-2020-compared-to-2019/

------------------------------------------------------------------------------------------------------------------------

**Kaggle Notebook:**
- [Kaggle.com: rosaaestrada - Mental Health & Age Analysis Among U.S. Adults](https://www.kaggle.com/code/rosaaestrada/mental-health-age-analysis-among-u-s-adults/edit/run/185611886)

------------------------------------------------------------------------------------------------------------------------

### Final Evaluation
**Person's Chi-Square:**
- Age and Veteran Status: The statistics and expected frequencies indicate that there is a significant relationship between age and veteran status with (p-value = 2.25e-227).
- Age and General Health: A significant relationship exists between age group and general health (p-value = 0.0), suggesting age plays a role in general health status.
- Age and Education: There is a statistically significant relationship between age group and education level (p-value = 3.24e-23), indicating age influences education.
- Mental Health and Veteran Status: A significant relationship is found between mental health and veteran status (p-value = 9.31e-14).
- Mental Health and Age: There is a significant relationship between mental health and age (p-value = 5.32e-157). Suggesting that different age groups experience varying levels of mental health, with some age groups more prone to "no good" mental health days than others.
- Mental Health and General Health: The relationship between mental health and general health is significant (p-value = 0.0).
- Mental Health and Education: There is a significant relationship between mental health and education (p-value = 5.21e-122).

**Analysis of Variance (ANOVA)**
- The ANOVA results indicate no significant differences across the different variables (Veteran, Age_Group, GeneralHealth, and Education) with respect to MentalHealth (all p-values > 0.05). This suggests that none of the individual factors considered in the analysis (Veteran, Age_Group, GeneralHealth, and Education) have a significant impact on mental health days when considered independently.

*Possible Explanations:*
- This could be because mental health is a complex and multifaceted issue, possibly influenced by the interplay of multiple factors rather than any single one. There could be other factors not considered in this analysis that may have a stronger impact.
- The effects of the variables might be intertwined, such that their combined impact on mental health is more significant than their individual contributions. This could lead to a lack of clear differences across individual variables.
- The sample size and distribution across the different categories may not have been large enough to detect small differences in mental health outcomes across the different groups.

*Future Directions:*
- Future research could explore the interaction effects between different variables, such as examining how combinations of age, education, general health, and veteran status impact mental health.
- Consider expanding the data collection to include additional factors such as income, employment status, or access to healthcare, which might influence mental health outcomes.

**Correlation Matrix between MentalHealth and the Independent Variables:**
- Age_Group_55 and MentalHealth (0.11): Shows a moderate correlation, suggesting that individuals in the age group 55 and older may have a slightly higher likelihood of experiencing "no good" mental health days.
- Age_Group_18_to_34 and MentalHealth (0.08): Shows a weaker correlation, suggesting that younger individuals in this age group may also experience some "no good" mental health days, but not as significantly as the older age group.
- GeneralHealth_Poor and MentalHealth (0.36): Indicates a strong positive relationship between individuals who report "poor" general health and the frequency of "no good" mental health days.
  - This correlation suggests that individuals with poor general health are more likely to experience more frequent "no good" mental health days.
-  GeneralHealth_Excellent and MentalHealth (0.27): Suggests that individuals who report excellent general health are less likely to experience "no good" mental health days. However, the correlation is not as strong as for those with poor general health.
-  Education_College and MentalHealth (0.10) and Education_High_School and MentalHealth (0.09): Suggests a slight association between education level and mental health days, with lower levels of education not potentially linked to more "no good" mental health days.
-  Veteran_No and Veteran_Yes with MentalHealth (0.03): Both show very weak correlations with MentalHealth. This suggests that veteran status does not have a strong impact on the likelihood of experiencing "no good" mental health days.

Overall, ages 55+ and 18-34 experience more "no good" mental health days compared to the middle-aged group. Whereas general health plays a significant role, with those in "poor" health experiencing more "no good" mental health days. Lastly, veteran status and education appear to have relatively weaker relationships with mental health outcomes.

------------------------------------------------------------------------------------------------------------------------

🟣 **Research Question:**
What ages experience the most "no good" mental health days?
- Individuals in the age group 55+ experience the most "no good" mental health days, as they have a moderately stronger positive correlation (0.11) with poor mental health days compared to the age group 18-34 with a weaker correlation (0.08). Indicating that while younger individuals may experience some "no good" mental health days, the impact is not as significant as it is for older individuals. Additionally, the relationship between age and general health may further impact mental health outcomes.

🟣 **Null Hypothesis (H0):**
Suggests there is no significant difference in the number of 'no good' mental health days reported across different age groups.
- We reject the Null Hypothesis (H0), as there is evidence suggesting a significant difference in 'no good' mental health days across different age groups.

🟣 **Alternative Hypothesis (H1):**
Suggests there is a significant difference in the number of 'no good' mental health days reported across different age groups.
- We accept the Alternative Hypothesis (H1), as there is evidence that there is a significant difference of 'no good' mental health days across different age groups.

🟢 Overall, age appears to be a critical factor in experiencing "no good" mental health days, particularly in age groups 55+ and 18-34. General health also plays a role in the number of "no good" mental health days across different age groups.

**As stated at the beginning of the project:**
The National Alliance on Mental Health Illness found that 50% of all lifetime mental illnesses begin at the age of 14 and 75% begin at the age of 24 (2023).
- These statistics suggest that mental health issues often begin at a young age, which aligns with the findings in this project that the age group 18-34 experiences a high likelihood of "no good" mental health days. It emphasizes the importance of early intervention and targeted support for younger individuals.

Understanding that mental health issues often begin in adolescence or early adulthood and can worsen in older age groups can help guide targeted interventions for these vulnerable populations. The increasing rate of mental health issues in recent years highlights the need for ongoing research and action to address underlying causes and provide adequate care and support. The correlation data suggests older individuals (55+) are more likely to experience "no good" mental health days, while younger individuals (18-34) also experience some negative mental health days. This aligns with The National Alliance on Mental Health Illness's findings regarding the onset of mental health issues beginning at a young age.
