#import libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from datetime import datetime

# functions for styling the negative values(check it out later)
def style_negative(v, props=''):
    """ Style negative values in the dataframe """
    try:
        return props if v<0 else None
    except:
        pass

# functions for styling the positive values
def style_positive(v, props=''):
    """ Style positive values in the dataframe """
    try:
        return props if v>0 else None
    except:
        pass

#creating a function to distinguish between countries USA,INDIA,OTHERS
def audience_simple(country):
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else : return 'Others' 

#importing data
@st.cache
# st.cache optimizes the time it takes for the reruns between a calculation through the cached function is only being 
# executed once, and every time after that you're actually hitting the cache.

def load_data():
    #iloc[1:,:] skips the first row and then includes everything else
    df_agg= pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:]
    #renaming the columns
    df_agg.columns = ['Video','Video title','Video publish time','Comments added','Shares','Dislikes','Likes',
                        'Subscribers lost','Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed','Average view duration',
                        'Views','Watch time (hours)','Subscribers','Your estimated revenue (USD)','Impressions','Impressions ctr(%)']
    # Utilizing the Date time converting from a string to actual usable format
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'])
    #Average view time duration converting from a string to actual usable format
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_agg['Engagement_ratio'] =  (df_agg['Comments added'] + df_agg['Shares'] +df_agg['Dislikes'] + df_agg['Likes']) /df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending = False, inplace = True) 
    df_agg_sub= pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments= pd.read_csv('Aggregated_Metrics_By_Video.csv')
    df_time=pd.read_csv('Video_Performance_Over_Time.csv')
    df_time['Date']=pd.to_datetime(df_time['Date'])
    return df_agg,df_agg_sub,df_comments,df_time

#create dataframes from the function
df_agg, df_agg_sub ,df_comments, df_time = load_data()

#feature engineering

#getting aggregated differential for all the data
df_agg_diff= df_agg.copy()

#setting up a 12 month filter
metric_data_12_mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)

#getting the median for every column
median_agg= df_agg_diff[df_agg_diff['Video publish time'] >= metric_data_12_mo].median()

#create differences from the median for values
#numeric columns
numeric_cols= np.array((df_agg_diff.dtypes == 'float64') |(df_agg_diff.dtypes == 'int64'))
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)

#merge daily data with publish data to get the delta (after setting the horizontal bar plot for the individual video )
df_time_diff = pd.merge(df_time,df_agg.loc[:,['Video','Video publish time']],left_on ='External Video ID',right_on ='Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

#get the data for last 12 months for a video
date_12_mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12) #DateOffset not it down
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12_mo]

# get_daily view data for the first 30 days along with its medians and percentile
view_days= pd.pivot_table(df_time_diff_yr,index = 'days_published',values ='Views',aggfunc = [np.mean,np.median,lambda x:np.percentile(x,80),lambda x:np.percentile(x,20)]).reset_index() 
view_days.columns=['days_published','mean_views','median_views','80pct_views','20pct_views']
view_days=view_days[view_days['days_published'].between(0,30)] #between note it down
views_cumulative=view_days.loc[:,['days_published','median_views','80pct_views','20pct_views']]
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()








#building Dashboard
#adding sidebar to the app that will display Aggregate metrics and Individual Video Analysis 
# after clicking on the option of Aggregate or Individual Video
add_sidebar= st.sidebar.selectbox('Aggregate or Individual Video',('Aggregate Metrics','Individual Video Analysis'))

#adding the Relevant metrics and delta
# Delta shows the % change in all the relevant metrics
if add_sidebar == 'Aggregate Metrics':
    st.write('Aggregate Data')
    df_agg_metrics=df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() -pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() -pd.DateOffset(months=12)
#median of all the aggregated metrics filtered by 6mo or 12mo
    metric_median_6mo= df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median()
    metric_median_12mo= df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median()
#creating the metrics for aggregate videos using columns in streamlit
    col1,col2,col3,col4,col5 = st.columns(5)
    columns=[col1,col2,col3,col4,col5]

#for loop for creating the metrics and displaying the delta with them
    count = 0
    for i in metric_median_6mo.index:
        with columns[count]:
            delta=(metric_median_6mo[i] - metric_median_12mo[i])/metric_median_12mo[i]
            st.metric(label=i, value=round(metric_median_6mo[i],1),delta = "{:.2%}".format(delta))
            count +=1 #this ensures that there are  
            if count >=5:
                count=0 



#creating the dataframe table under the metrics displayed 
#creating the date column for the data frame  
df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x:x.date())
#adding the columns to the dataframe 
df_agg_diff_final = df_agg_diff.loc[:,['Video title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']] 

#gets a list of the aggregated data
df_agg_numeric_list = df_agg_diff_final.median().index.tolist()
#creating a dictionary to pass in the names of the columns and then passing the percent format converting the metrics of the dataframe 
# into percentages
df_to_pct ={}
for i in df_agg_numeric_list:
    df_to_pct[i] = '{:.1%}'.format      #tells the format in which we want the information to be displayed

#creating the dataframe in streamlit (check it out later)
#now we come to formatting the dataframe and one of the problems is that we don't want to format the whole dataframe but 
#only a number of columns
st.dataframe(df_agg_diff_final.style.applymap(style_negative,props= 'color:red;').applymap(style_positive,props= 'color:green;').format(df_to_pct))

#selecting the title of the video for individual in depth analysis 
if add_sidebar == 'Individual Video Analysis':     #adds the sidebar
    videos= tuple(df_agg['Video title'])           #brings all the titles together in a tuple 
    video_select= st.selectbox('Select a video',videos)  #adds all the titles to the selectbox


    #displaying the data of the video that is selected by the user
    agg_filtered = df_agg[df_agg['Video title'] == video_select] #for the aggregated data
    agg_sub_filtered =df_agg_sub[df_agg_sub['Video Title'] == video_select] #for the data of the subscribers
    agg_sub_filtered['Country'] = df_agg_sub['Country Code'].apply(audience_simple) #for the countries
    agg_sub_filtered.sort_values('Is Subscribed',inplace=True) #only showing the data from the subscribed audiences

    #drawing the charts
    #Plotly express chart

    fig=px.bar(agg_sub_filtered,x ='Views',y ='Is Subscribed', color ='Country',orientation = 'h')
    #             data           x axis      y axis              color will be    horizontal
    #                                                            country wise     orientation
    st.plotly_chart(fig)  #passing the fig plot on the streamlit dashboard

    #data for the percentile chart for 30 days
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 =first_30.sort_values('days_published')

    fig2 = go.Figure()
    #adding percentile dash lines so that we know where our video fits in
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=views_cumulative['20pct_views'],
                mode = 'lines',
                name='20th percentile',line=dict(color='purple',dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=views_cumulative['median_views'],
                mode = 'lines',
                name='50th percentile',line=dict(color='black',dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=views_cumulative['80pct_views'],
                mode = 'lines',
                name='80th percentile',line=dict(color='royal blue',dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'],y=first_30['Views'].cumsum(),
                mode = 'lines',
                name='Current Video',line=dict(color='firebrick',width=8)))

    fig2.update_layout(title='view comparisons for the first 60 days',
                    xaxis_title='Days since published',
                    yaxis_title='Cumulative views'
                    )

    st.plotly_chart(fig2)

