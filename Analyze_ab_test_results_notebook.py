#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[113]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[114]:


#load the data and show a sample of it
df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[115]:


#get the number of rows
number_of_rows = df.shape[0]
number_of_rows


# c. The number of unique users in the dataset.

# In[116]:


#get the number of the unique users
unique_users = df.user_id.unique()
len(unique_users)


# d. The proportion of users converted.

# In[117]:


#get the proportion of the users who converted
df[df['converted'] == 1]['user_id'].count()/df.converted.count()


# e. The number of times the `new_page` and `treatment` don't match.

# In[118]:


#part of dataframe that group is tratment and landingpage is not newpage
    
df_A = df.query('group == "treatment" & landing_page != "new_page"')

#part of dataframe that group is  not tratment and landingpage is newpage

df_B = df.query('group != "treatment" & landing_page == "new_page"') 


#number of times that new_page and treatment does not match

number_of_times = (df_A.count()+df_B.count())['timestamp']           
number_of_times


# f. Do any of the rows have missing values?

# In[119]:


#see if there is any null values in the dataset
df.info()   


# - there is no missing values in the data

# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[120]:


df1 = df.drop(df[(df['group'] == "treatment") & (df['landing_page'] != "new_page")].index)
df2 = df1.drop(df1[(df['group'] =="control") & (df1['landing_page'] != "old_page")].index)


# In[121]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[122]:


#check the number of nuique user id
uniquedf2 = df2.user_id.unique()
len(uniquedf2)


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[123]:


#get the duplicated row user_id
repeated_user_id = df2[df2['user_id'].duplicated() == True]['user_id']
repeated_user_id


# c. What is the row information for the repeat **user_id**? 

# In[124]:


#row information of the repeated user_id
df2[df2['user_id'].duplicated() == True] 


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[125]:


#drop the duplicated row
df2 = df2.drop(labels = 2893)


# In[126]:


#make sure that there is no duplicated rows
df2.duplicated().sum()


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[127]:


#get the probability of converted individualls
df2[df2['converted'] == 1]['user_id'].count() / df2.converted.count()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[128]:


#the probability of indviduals which they are in control group
prob_converted = df2.groupby('group')['converted'].mean()
prob_converted.control


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[129]:


#the probability of indviduals which they are in treatment group
prob_converted.treatment


# d. What is the probability that an individual received the new page?

# In[130]:


#probability of who recievedthe new page 
df2[df2['landing_page'] == 'new_page']['user_id'].count() / df2.landing_page.count()


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Your answer goes here.**
# 
# I think no ! , there is not sufficient evidence to say that the new treatment page leads to more conversions.
# 
# we can see that in the test, Half of the population received the old_page and half of the population received the new_page. The population is considerable in size (290584 users).
# 
# there is about 12.04% that received the old_page were converted. 11.88% that received the new_page were converted. so to sum up, the new_page can not do any  increasing the conversion rate.
# 
# 
# 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Put your answer here.**
# 
# we can say !
# 
# Null hypothesis: the conversion rate of the old_page is greater or the same than the conversion rate of the newpage.p(old) >= p(new)
# Alternative hypothesis: the conversion rate of the old_page is less than the conversion rate of the newpage. p(old)< p(new)

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[131]:


#hint: we will use the whole dataset because the null hypothesis states that the conversion rate of the old_page and new_page is the same
p_new = df2.converted.mean()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[132]:


# it is the same as new_page
p_old = df2.converted.mean() 
p_old 


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[133]:


n_new = df2[df2['group'] == 'treatment']['user_id'].count()
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[134]:


n_old = df2[df2['group'] == 'control']['user_id'].count()
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[135]:



new_page_converted = np.random.binomial(1,p_new,n_new)
new_page_converted.mean()


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[136]:



old_page_converted = np.random.binomial(1,p_old,n_old)
old_page_converted.mean()


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[137]:


diff  = new_page_converted.mean() - old_page_converted.mean()
diff


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[138]:



p_diffs = []
new_converted_simulation = np.random.binomial(n_new, p_new, 10000)/n_new
old_converted_simulation = np.random.binomial(n_old, p_old, 10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation

    


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[139]:


plt.hist(p_diffs)
plt.axvline(diff , color = 'r')


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[140]:


exp_prop = (df2[df2['group'] == 'treatment']['converted'].mean())
cont_prop = (df2[df2['group'] == 'control']['converted'].mean())
diff = exp_prop - cont_prop
diff


# In[141]:


#the proportion of p_diffs greater than the observe difference
p_diffs = np.array(p_diffs)
(p_diffs > diff).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Put your answer here.**
# 
#  0.903 is called scientifically p-value, which determines the probability of obtaining our observed statistic (or one more extreme in favor of the alternative) if the null hypothesis is true.
# 
# This value means that we cannot reject the null hypothesis and that we do not have sufficient evidence that the new_page has a higher conversion rate than the old_page.
# 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[142]:


import statsmodels.api as sm

convert_old = sum(df2[df2['group'] == 'control']['converted'])
convert_new = sum(df2[df2['group'] == 'treatment']['converted'])
n_old = df2[df2['landing_page'] == 'old_page']['user_id'].count()
n_new = df2[df2['landing_page'] == 'new_page']['user_id'].count()


# In[143]:


convert_new, convert_old


# In[144]:


n_old , n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[145]:


z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old],value=None, alternative='larger', prop_var=False)

z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Put your answer here.**
# 
# - A negative z-score and the value of p-value state that we can't  reject the null hypothesis
# - The Null being the converted rate of the old_page is the same or greater than the converted rate of the new_page
# 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Put your answer here.**
# 
# we will use logistic regression as it is used in states that we work on true or false binary classification (converted 1 or non converted 0)

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[146]:


df2['intersept'] = 1


# In[147]:


df2[['treatment' ,'control']] = pd.get_dummies(df2['group'])
df2 = df2.drop('control',axis = 1)
df2.rename(columns = {'treatment' : 'ab_page'} , inplace = True)
df2 = df2.drop(columns=['converted']).assign(converted= df2['converted'])

df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[148]:



log_reg = sm.Logit(df2['converted'] , df2[['intersept' , 'ab_page']])
fit_log = log_reg.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[149]:


fit_log.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **Put your answer here.**
# 
# 
# The p-value here of the ab_page column is 0.1899 and we can see that it is lower than the p-value in part ll using the z-score function. The main reason of the difference between the p-values for Part II and the Regression Model is due to the intercept added.
# 
# this is a two-sided t-test compared to a one-sided t-test in part II

# - Hypothesis for null is H0 : p_new−p_old= 0
#  
# - Hypothesis for alternative is H1  : p_new−p_old ! = 0
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Put your answer here.**
# 
# - I think we should consider other factors in order to identify other potencial influences on the conversion rate.
#   but the disadvantage of it is that the model will be a bit complex.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[150]:


data_country = pd.read_csv('countries.csv')

df_new = data_country.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new['intercept'] = 1
### Create the  dummy variables
df_new[['CA','UK', 'US']]= pd.get_dummies(df_new['country'])

df_new.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[151]:


fit_log = sm.Logit(df_new['converted'],df_new[['intercept','ab_page','CA','US']])
results = fit_log.fit()
results.summary2()


# ##### Summary and conclusion on regression
# 
# The p_value for both interaction terms is higher than 0.05.
# 
# Thus, the influence of landing_page in the US is not different to the influence of landing_page in the other countries.
# 
# And the influence of landing_page in Canada is not different to the influence of landing_page in the other countries.
# 
# 
# ###### Conclusions
# In conclusion, there is not enough evidence that the new_page increases the conversion rate as compared to the old_page. This is based on the probability figures, A/B testand regression. There is no strong evidence that the countries (US, CA and UK) influence the conversion rate.
# 
# Since the sample size is large continuing the testing of the new_page is likely not necessary. It is best to focus on the development of another new landing page.

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

