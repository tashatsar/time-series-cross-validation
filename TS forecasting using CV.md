# Table of contents


* [Imports, data reading and descriptive analysis](#imp)
    * [Missing values imputation](#miss)

* [Best model selection (TS CV)](#best)
    * ["Complete" time series](#comp)
    * ["Short" time series](#short)

* [Forecasting and conclusions](#forecast)

## Imports, data reading and descriptive analysis <a class="anchor" id="imp"></a>



# First Markdown Cell
* [Notebook](test.ipynb)
* [Markdown](test.md)
* [Text](test.txt)
* [Cell](#Second-Markdown-Cell)
* [HTML anchor](#intro)
* [Web](https://stackoverflow.com)

<ul>
<li><a href="#intro">HTML anchor in HTML</a></li>
</ul>

## Second Markdown Cell

<a id="intro"></a>
## Introduction





```python
import numpy as np
import pandas as pd

import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import sys
sys.path.append(os.getcwd())
```


```python
from ts_cv import ts_cv_metrics
from wrappers import WrapperAutoArima, WrapperProphet, WrapperHoltWinters
```


```python
df = pd.read_csv('train.csv')
df['dt'] = pd.to_datetime(df['dt'])
df.columns = ['id', 'ds', 'y']
df.groupby('id').agg({'ds': ['min', 'max', 'count'], 'y': ['mean', 'std']})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">ds</th>
      <th colspan="2" halign="left">y</th>
    </tr>
    <tr>
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>453</td>
      <td>776.880353</td>
      <td>192.248316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>453</td>
      <td>485.492274</td>
      <td>122.866411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>448</td>
      <td>595.209598</td>
      <td>108.123227</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>448</td>
      <td>435.170982</td>
      <td>148.888847</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>385.991156</td>
      <td>54.456945</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>454.067849</td>
      <td>109.136251</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>640.762749</td>
      <td>166.423599</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>450</td>
      <td>634.192000</td>
      <td>115.375308</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>1549.520748</td>
      <td>297.530101</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>779.500443</td>
      <td>184.243248</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>675.758163</td>
      <td>104.065112</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>452</td>
      <td>513.794027</td>
      <td>135.294493</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>450</td>
      <td>776.747111</td>
      <td>283.941149</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>687.877211</td>
      <td>128.385238</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>450</td>
      <td>606.364222</td>
      <td>177.961598</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>453</td>
      <td>807.902428</td>
      <td>180.486750</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>446.958980</td>
      <td>97.033838</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>484.503741</td>
      <td>147.873661</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>448</td>
      <td>646.840625</td>
      <td>177.672308</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>453.589135</td>
      <td>135.575430</td>
    </tr>
  </tbody>
</table>
</div>



There are three groups of TS: 
* almost complete TS with 448-452 observations
* 'complete' TS with 453 observations
* short TS with less then 294 observations
    
The interesting thing is that start and end dates are the same of all of the TS, but the amount of days does not suit even the compltete TS so let's investigate further.


```python
df['day'] = df['ds'].dt.dayofweek
df['mnth'] = df['ds'].apply(lambda x: x.replace(day=1))
#number of days of week (0 - Sunday, 1 - Monday, etc.) for each store
df.pivot_table(values='y', index='id', columns='day',aggfunc='count')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>day</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>69</td>
      <td>8</td>
      <td>72</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72</td>
      <td>69</td>
      <td>8</td>
      <td>72</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48</td>
      <td>44</td>
      <td>7</td>
      <td>48</td>
      <td>50</td>
      <td>51</td>
      <td>46</td>
    </tr>
    <tr>
      <th>5</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>6</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>7</th>
      <td>72</td>
      <td>69</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>48</td>
      <td>44</td>
      <td>7</td>
      <td>48</td>
      <td>50</td>
      <td>51</td>
      <td>46</td>
    </tr>
    <tr>
      <th>9</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>10</th>
      <td>48</td>
      <td>44</td>
      <td>7</td>
      <td>48</td>
      <td>50</td>
      <td>51</td>
      <td>46</td>
    </tr>
    <tr>
      <th>11</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>77</td>
      <td>73</td>
    </tr>
    <tr>
      <th>12</th>
      <td>71</td>
      <td>70</td>
      <td>7</td>
      <td>74</td>
      <td>77</td>
      <td>78</td>
      <td>73</td>
    </tr>
    <tr>
      <th>13</th>
      <td>48</td>
      <td>44</td>
      <td>7</td>
      <td>48</td>
      <td>50</td>
      <td>51</td>
      <td>46</td>
    </tr>
    <tr>
      <th>14</th>
      <td>72</td>
      <td>69</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>15</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>73</td>
    </tr>
    <tr>
      <th>16</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>17</th>
      <td>48</td>
      <td>44</td>
      <td>7</td>
      <td>48</td>
      <td>50</td>
      <td>51</td>
      <td>46</td>
    </tr>
    <tr>
      <th>18</th>
      <td>72</td>
      <td>69</td>
      <td>7</td>
      <td>74</td>
      <td>77</td>
      <td>78</td>
      <td>71</td>
    </tr>
    <tr>
      <th>19</th>
      <td>72</td>
      <td>70</td>
      <td>8</td>
      <td>74</td>
      <td>78</td>
      <td>78</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivot_df = df.pivot_table(values='y', index=['id', 'mnth'], columns='day', aggfunc='count')
pivot_df = pivot_df.reset_index()
pivot_df.groupby('mnth').min()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>day</th>
      <th>id</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
    <tr>
      <th>mnth</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2016-02-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2016-03-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2016-04-01</th>
      <td>0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2016-05-01</th>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2016-06-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2016-07-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2016-08-01</th>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2016-09-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2016-10-01</th>
      <td>0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2016-11-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2016-12-01</th>
      <td>0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2017-02-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2017-03-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2017-04-01</th>
      <td>0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2017-05-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2017-06-01</th>
      <td>0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Lets exclude from consideration 01 and 02 months of 2016 as irrelevant becaue of different schedule of days off. Also let's impute missing dates.


```python
df = df[df['ds']>=pd.to_datetime('2016-03-01')]

dates_list = pd.date_range(df['ds'].min(), df['ds'].max())
dates_list = dates_list[dates_list.weekday != 2] # exclude second day of week (day off)

tmp_dict = pd.DataFrame(dates_list).copy()
tmp_dict.columns = ['ds']
for ts in range(20):
    tmp_dict['id'] = ts
    df = pd.merge(df, tmp_dict, how='outer')

df = df.sort_values(['id', 'ds'])

df['day'] = df['ds'].dt.dayofweek #day of week
df['mnth'] = df['ds'].apply(lambda x: x.replace(day=1)) #month
df = df.reset_index(drop=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ds</th>
      <th>y</th>
      <th>day</th>
      <th>mnth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2016-03-01</td>
      <td>564.2</td>
      <td>1</td>
      <td>2016-03-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2016-03-03</td>
      <td>1161.1</td>
      <td>3</td>
      <td>2016-03-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2016-03-04</td>
      <td>1007.0</td>
      <td>4</td>
      <td>2016-03-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2016-03-05</td>
      <td>804.5</td>
      <td>5</td>
      <td>2016-03-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2016-03-06</td>
      <td>887.1</td>
      <td>6</td>
      <td>2016-03-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8335</th>
      <td>19</td>
      <td>2017-06-25</td>
      <td>338.3</td>
      <td>6</td>
      <td>2017-06-01</td>
    </tr>
    <tr>
      <th>8336</th>
      <td>19</td>
      <td>2017-06-26</td>
      <td>358.4</td>
      <td>0</td>
      <td>2017-06-01</td>
    </tr>
    <tr>
      <th>8337</th>
      <td>19</td>
      <td>2017-06-27</td>
      <td>278.5</td>
      <td>1</td>
      <td>2017-06-01</td>
    </tr>
    <tr>
      <th>8338</th>
      <td>19</td>
      <td>2017-06-29</td>
      <td>739.1</td>
      <td>3</td>
      <td>2017-06-01</td>
    </tr>
    <tr>
      <th>8339</th>
      <td>19</td>
      <td>2017-06-30</td>
      <td>629.7</td>
      <td>4</td>
      <td>2017-06-01</td>
    </tr>
  </tbody>
</table>
<p>8340 rows × 5 columns</p>
</div>




```python
# some plots
short_ts = [4, 8, 10, 13, 17]

plt.figure(figsize=(12, 6))
for ts in short_ts:
    sns.lineplot(data=df.loc[df['id']==ts], x='ds', y='y')
plt.title('"Short" Time Series')

plt.figure(figsize=(12, 6))
for ts in [0, 5, 10]:
    sns.lineplot(data=df.loc[df['id']==ts], x='ds', y='y')
plt.title('Some of Time Series')
```




    Text(0.5, 1.0, 'Some of Time Series')




    
![png](output_9_1.png)
    



    
![png](output_9_2.png)
    



### Missing values imputation <a class="anchor" id="miss"></a>

Seems that TS have weekly seasonal component, almost no trend and some monthly changes within a year

#### Missing values characteristics

Thus, all of the stores have missing values of suppossively the following nature:
* days off on Monday for two first months of 2016 and on Tuesday for the rest of the time
* holidays with non-working days in April and May 
  
Also some stores have non working days in December and July, but not all of them. Five TS have a gap in data from 07 of 2016 until 01 of 2017.

#### General approach for imputation

In this task different methods/algorithms of TS forecasting were used and some of those models are sensitive not to dates but to the order of data. Because of that it's necessary to impute missing values. 

Unfortunately, Python does not have similar to R na_seadec functionality, so a simple approach for imputation of missing values with seasonality was used: weekly seasonality was calculated for each day of week within every store. 


```python
df_seas_coef = df.groupby(['id', 'day'], as_index=False).agg('median')

for ts in range(20):
    tmp = df_seas_coef[df_seas_coef['id']==ts].copy()
    tmp['coef'] = tmp['y']/tmp['y'].mean()
    df_seas_coef = pd.merge(df_seas_coef, tmp, how='outer')
```


```python
df = pd.merge(df, df_seas_coef[['id', 'day', 'coef']], how='left', on=['id', 'day'])
df = df[~df['coef'].isna()]
df['sunday'] = df['ds']-df['day'].apply(lambda x: datetime.timedelta(x)) 

df = pd.merge(df, 
              pd.DataFrame(df.groupby(['id', 'sunday'])['y'].mean()).rename(columns={'y':'y_mean'}).reset_index())

# weekly seasonality calculation
df['y_seas']= df['y'] 
df['y_seas'] = np.where(df['y_seas'].isna(), df['y_mean']*df['coef'], df['y_seas']).tolist()

# target with missing values
df['y_init'] = df['y'] 
df['y'] = df['y_seas'] 

#linear interpolation
df['y_lin'] = df['y'].interpolate(method='linear', axis=0)
df['y_lin'] = np.where(df['y_init'].isna(), df['y_lin']*df['coef'], df['y_init']).tolist()

# linear interpolation with seasonality 
df['y_lin_seas'] = df['y_lin']*df['coef']
df['y_lin_seas'] = np.where(df['y_init'].isna(), df['y_lin_seas']*df['coef'], df['y_init']).tolist()

# among seasonal, linear and linear-seasonal interpolation the best result shown was for the first one
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ds</th>
      <th>y</th>
      <th>day</th>
      <th>mnth</th>
      <th>coef</th>
      <th>sunday</th>
      <th>y_mean</th>
      <th>y_seas</th>
      <th>y_init</th>
      <th>y_lin</th>
      <th>y_lin_seas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2016-03-01</td>
      <td>564.2</td>
      <td>1</td>
      <td>2016-03-01</td>
      <td>0.682837</td>
      <td>2016-02-29</td>
      <td>884.780000</td>
      <td>564.2</td>
      <td>564.2</td>
      <td>564.2</td>
      <td>564.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2016-03-03</td>
      <td>1161.1</td>
      <td>3</td>
      <td>2016-03-01</td>
      <td>1.231378</td>
      <td>2016-02-29</td>
      <td>884.780000</td>
      <td>1161.1</td>
      <td>1161.1</td>
      <td>1161.1</td>
      <td>1161.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2016-03-04</td>
      <td>1007.0</td>
      <td>4</td>
      <td>2016-03-01</td>
      <td>1.128510</td>
      <td>2016-02-29</td>
      <td>884.780000</td>
      <td>1007.0</td>
      <td>1007.0</td>
      <td>1007.0</td>
      <td>1007.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2016-03-05</td>
      <td>804.5</td>
      <td>5</td>
      <td>2016-03-01</td>
      <td>0.961146</td>
      <td>2016-02-29</td>
      <td>884.780000</td>
      <td>804.5</td>
      <td>804.5</td>
      <td>804.5</td>
      <td>804.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2016-03-06</td>
      <td>887.1</td>
      <td>6</td>
      <td>2016-03-01</td>
      <td>1.030313</td>
      <td>2016-02-29</td>
      <td>884.780000</td>
      <td>887.1</td>
      <td>887.1</td>
      <td>887.1</td>
      <td>887.1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8335</th>
      <td>19</td>
      <td>2017-06-25</td>
      <td>338.3</td>
      <td>6</td>
      <td>2017-06-01</td>
      <td>0.968778</td>
      <td>2017-06-19</td>
      <td>364.733333</td>
      <td>338.3</td>
      <td>338.3</td>
      <td>338.3</td>
      <td>338.3</td>
    </tr>
    <tr>
      <th>8336</th>
      <td>19</td>
      <td>2017-06-26</td>
      <td>358.4</td>
      <td>0</td>
      <td>2017-06-01</td>
      <td>0.937321</td>
      <td>2017-06-26</td>
      <td>501.425000</td>
      <td>358.4</td>
      <td>358.4</td>
      <td>358.4</td>
      <td>358.4</td>
    </tr>
    <tr>
      <th>8337</th>
      <td>19</td>
      <td>2017-06-27</td>
      <td>278.5</td>
      <td>1</td>
      <td>2017-06-01</td>
      <td>0.622322</td>
      <td>2017-06-26</td>
      <td>501.425000</td>
      <td>278.5</td>
      <td>278.5</td>
      <td>278.5</td>
      <td>278.5</td>
    </tr>
    <tr>
      <th>8338</th>
      <td>19</td>
      <td>2017-06-29</td>
      <td>739.1</td>
      <td>3</td>
      <td>2017-06-01</td>
      <td>1.299883</td>
      <td>2017-06-26</td>
      <td>501.425000</td>
      <td>739.1</td>
      <td>739.1</td>
      <td>739.1</td>
      <td>739.1</td>
    </tr>
    <tr>
      <th>8339</th>
      <td>19</td>
      <td>2017-06-30</td>
      <td>629.7</td>
      <td>4</td>
      <td>2017-06-01</td>
      <td>1.128331</td>
      <td>2017-06-26</td>
      <td>501.425000</td>
      <td>629.7</td>
      <td>629.7</td>
      <td>629.7</td>
      <td>629.7</td>
    </tr>
  </tbody>
</table>
<p>8340 rows × 12 columns</p>
</div>



--- 

## Best model selection using time series cross-validation <a class="anchor" id="best"></a>


The first two models are sensitive not to dates but to the order of data while the last one takes into consideration information of dates and days of week. Because of that it's necessary to impute missing values at least for Holt-Winters and ARIMA models.

In this task we are going to use different methods of TS forecasting:
* Holt-Winters model with additive trend and seasonality: 
    * no tuning of parameters, seasonality supposed to be weekly (with cycle of 6 because Tuesday is always missing
    
    
* ARIMA model with auto tuning of parameters:
    * the order (or number of time lags) of the auto-regressive (“AR”) model **p** varies between 1 and 4
    * the order of the moving-average (“MA”) model **q** varies between 1 and 4
    * the order of the auto-regressive portion of the seasonal model **P** varies between 1 and 4
    * the order of the moving-average portion of the seasonal mode **Q** varies between 1 and 4
    * the maximum number of non-seasonal differences **d** varies between 1 and 4
    * the order of the seasonal differencing **D** varies between 1 and 5


* facebook Prophet model with tuning of:
    * seasonality_mode - additive and multiplicative
    * weekly_seasonality - witout weekly seasonality and with one using a Fourier order 2, 3, 5
    * monthly_seasonality - without and with using a Fourier order 2, 3
    * yearly_seasonality was set to false due to too short TS
    
For Prophet model the best set of parameters was choosen according to the sMAE metric value on three fold time series cross-validation. cross-validation for parameters selection is always performed only at train dataset.        
        

### "Complete" time series <a class="anchor" id="comp"></a>

The best model for forecasting "complete" time series (the majority of all TS, the ones that have data witout big interruptions) was choosen using three folds time series cross-validation with an algorithm of rolling window. Thus, for each of time series considered there was a forecast made for three independent time periods with duration of 31 day. 


```python
complete_ts = [0, 1, 15, 2, 3, 5, 6, 7, 9, 11, 12, 14, 16, 18, 19]
short_ts = [4, 8, 10, 13, 17]
```


```python
hw_model = WrapperHoltWinters()
arima_model = WrapperAutoArima()
prophet_model = WrapperProphet()
models = [hw_model, arima_model, prophet_model]

result = []
for ts in tqdm(complete_ts):
    try:
        tmp = df.loc[df['id']==ts]
        for model in models:
            mean_smae, mean_mape, model_name = ts_cv_metrics(tmp, model, cv_folds=3, horizon=31)
            res = {'id': ts, 'model': model_name,
                   'smae': mean_smae, 'mape': mean_mape}
            result.append(res)
    except:
        print(ts, 'smth went wrong')
```


```python
result = pd.DataFrame(result)
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>model</th>
      <th>smae</th>
      <th>mape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>HoltWinters</td>
      <td>0.207969</td>
      <td>0.220126</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>AutoArima</td>
      <td>0.173781</td>
      <td>0.179067</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Prophet</td>
      <td>0.129320</td>
      <td>0.129346</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>HoltWinters</td>
      <td>0.185294</td>
      <td>0.185153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>AutoArima</td>
      <td>0.171511</td>
      <td>0.161834</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>Prophet</td>
      <td>0.130690</td>
      <td>0.124025</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>HoltWinters</td>
      <td>0.174094</td>
      <td>0.178552</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15</td>
      <td>AutoArima</td>
      <td>0.167237</td>
      <td>0.165050</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15</td>
      <td>Prophet</td>
      <td>0.115405</td>
      <td>0.108300</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>HoltWinters</td>
      <td>0.148177</td>
      <td>0.143284</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>AutoArima</td>
      <td>0.140365</td>
      <td>0.132862</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>Prophet</td>
      <td>0.098743</td>
      <td>0.094825</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3</td>
      <td>HoltWinters</td>
      <td>0.248699</td>
      <td>0.266441</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>AutoArima</td>
      <td>0.243386</td>
      <td>0.265810</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>Prophet</td>
      <td>0.145481</td>
      <td>0.147477</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>HoltWinters</td>
      <td>0.234411</td>
      <td>0.255255</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>AutoArima</td>
      <td>0.199739</td>
      <td>0.206200</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>Prophet</td>
      <td>0.156101</td>
      <td>0.167875</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6</td>
      <td>HoltWinters</td>
      <td>0.203005</td>
      <td>0.213780</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>AutoArima</td>
      <td>0.200730</td>
      <td>0.201518</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6</td>
      <td>Prophet</td>
      <td>0.155466</td>
      <td>0.164282</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>HoltWinters</td>
      <td>0.145933</td>
      <td>0.136509</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>AutoArima</td>
      <td>0.136542</td>
      <td>0.129567</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>Prophet</td>
      <td>0.091509</td>
      <td>0.087947</td>
    </tr>
    <tr>
      <th>24</th>
      <td>9</td>
      <td>HoltWinters</td>
      <td>0.196656</td>
      <td>0.190611</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9</td>
      <td>AutoArima</td>
      <td>0.177318</td>
      <td>0.176246</td>
    </tr>
    <tr>
      <th>26</th>
      <td>9</td>
      <td>Prophet</td>
      <td>0.112773</td>
      <td>0.109840</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11</td>
      <td>HoltWinters</td>
      <td>0.213008</td>
      <td>0.223455</td>
    </tr>
    <tr>
      <th>28</th>
      <td>11</td>
      <td>AutoArima</td>
      <td>0.214573</td>
      <td>0.214508</td>
    </tr>
    <tr>
      <th>29</th>
      <td>11</td>
      <td>Prophet</td>
      <td>0.138155</td>
      <td>0.143233</td>
    </tr>
    <tr>
      <th>30</th>
      <td>12</td>
      <td>HoltWinters</td>
      <td>0.259463</td>
      <td>0.273931</td>
    </tr>
    <tr>
      <th>31</th>
      <td>12</td>
      <td>AutoArima</td>
      <td>0.249863</td>
      <td>0.256040</td>
    </tr>
    <tr>
      <th>32</th>
      <td>12</td>
      <td>Prophet</td>
      <td>0.143384</td>
      <td>0.144366</td>
    </tr>
    <tr>
      <th>33</th>
      <td>14</td>
      <td>HoltWinters</td>
      <td>0.177329</td>
      <td>0.170061</td>
    </tr>
    <tr>
      <th>34</th>
      <td>14</td>
      <td>AutoArima</td>
      <td>0.223716</td>
      <td>0.226113</td>
    </tr>
    <tr>
      <th>35</th>
      <td>14</td>
      <td>Prophet</td>
      <td>0.135872</td>
      <td>0.134552</td>
    </tr>
    <tr>
      <th>36</th>
      <td>16</td>
      <td>HoltWinters</td>
      <td>0.178969</td>
      <td>0.178907</td>
    </tr>
    <tr>
      <th>37</th>
      <td>16</td>
      <td>AutoArima</td>
      <td>0.157334</td>
      <td>0.144577</td>
    </tr>
    <tr>
      <th>38</th>
      <td>16</td>
      <td>Prophet</td>
      <td>0.122563</td>
      <td>0.114991</td>
    </tr>
    <tr>
      <th>39</th>
      <td>18</td>
      <td>HoltWinters</td>
      <td>0.201371</td>
      <td>0.201734</td>
    </tr>
    <tr>
      <th>40</th>
      <td>18</td>
      <td>AutoArima</td>
      <td>0.198999</td>
      <td>0.186070</td>
    </tr>
    <tr>
      <th>41</th>
      <td>18</td>
      <td>Prophet</td>
      <td>0.137235</td>
      <td>0.138189</td>
    </tr>
    <tr>
      <th>42</th>
      <td>19</td>
      <td>HoltWinters</td>
      <td>0.203757</td>
      <td>0.199414</td>
    </tr>
    <tr>
      <th>43</th>
      <td>19</td>
      <td>AutoArima</td>
      <td>0.207571</td>
      <td>0.196393</td>
    </tr>
    <tr>
      <th>44</th>
      <td>19</td>
      <td>Prophet</td>
      <td>0.137634</td>
      <td>0.133933</td>
    </tr>
  </tbody>
</table>
</div>



### "Short" time series<a class="anchor" id="short"></a>

For incomplete time series the same approach as for the rest of data was used. The initial idea is that we can forecast using as a train sample only short part of a dataset, which is the closest to the forecasted period of time. Two fold cross-validation was used due to the length of the time series.


```python
hw_model = WrapperHoltWinters()
arima_model = WrapperAutoArima()
prophet_model = WrapperProphet()
models = [hw_model, arima_model, prophet_model]

result_short = []
for ts in tqdm(short_ts):
    try:
        tmp = df.loc[(df['id']==ts)&(df['ds']>=pd.to_datetime('2017-01-02'))]
        for model in models:
            mean_smae, mean_mape, model_name = ts_cv_metrics(tmp, model, cv_folds=2, horizon=31)
            res = {'id': ts, 'model': model_name,
                   'smae': mean_smae, 'mape': mean_mape}
            result_short.append(res)
    except:
        print(ts, 'smth went wrong')        
```


```python
result_short = pd.DataFrame(result_short)
result_short
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>model</th>
      <th>smae</th>
      <th>mape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>HoltWinters</td>
      <td>0.233255</td>
      <td>0.239264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>AutoArima</td>
      <td>0.182528</td>
      <td>0.200218</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Prophet</td>
      <td>0.157901</td>
      <td>0.171696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>HoltWinters</td>
      <td>0.163539</td>
      <td>0.156702</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>AutoArima</td>
      <td>0.137863</td>
      <td>0.135526</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>Prophet</td>
      <td>0.117764</td>
      <td>0.119826</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>HoltWinters</td>
      <td>0.149362</td>
      <td>0.150714</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>AutoArima</td>
      <td>0.113196</td>
      <td>0.113814</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>Prophet</td>
      <td>0.098845</td>
      <td>0.103349</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13</td>
      <td>HoltWinters</td>
      <td>0.129837</td>
      <td>0.120907</td>
    </tr>
    <tr>
      <th>10</th>
      <td>13</td>
      <td>AutoArima</td>
      <td>0.124352</td>
      <td>0.120898</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>Prophet</td>
      <td>0.088696</td>
      <td>0.091388</td>
    </tr>
    <tr>
      <th>12</th>
      <td>17</td>
      <td>HoltWinters</td>
      <td>0.213321</td>
      <td>0.224631</td>
    </tr>
    <tr>
      <th>13</th>
      <td>17</td>
      <td>AutoArima</td>
      <td>0.198158</td>
      <td>0.193846</td>
    </tr>
    <tr>
      <th>14</th>
      <td>17</td>
      <td>Prophet</td>
      <td>0.133809</td>
      <td>0.136719</td>
    </tr>
  </tbody>
</table>
</div>



## Forecasting and conclusions<a class="anchor" id="forecast"></a>

Thus, for all of the TS the best result according to the key performance metric sMAE is achieved using Prophet model from Facebook, even thoough in some of cases AutoArima was also giving good results. 

Pipeline used:
1. Descriptive analysis, discovery of days-off, holidays, missing values and even missing time periods. 
2. Missing values imputation using weekly seasonality, division of time series into two groups: complete time series with big amount of observations and short time series (25% of all TS) with missing period in 2016 year. 
2. Testing on time series cross-validation three different models: Holt-Winters, auto ARIMA and Prophet for two groups of time series.
3. Forecasting for 07 month of 2017 using the best model selected at the previous step: Prophet.  


```python
prophet_model = WrapperProphet()
forecast_result = pd.DataFrame()
for ts in tqdm(range(20)):
    if ts in short_ts:
        tmp = df.loc[(df['id']==ts)&(df['ds']>=pd.to_datetime('2017-01-02'))]
        cv=3
    else: 
        tmp = df.loc[df['id']==ts]
        cv=5
    prophet_model.fit(tmp, cv=cv)
    pred = prophet_model.predict(tmp, 31)
    pred['target'] = np.where(pred['ds'].dt.dayofweek == 2, 0, pred['yhat']).tolist()
    pred['id'] = ts
    pred = pred[['id', 'ds', 'target']]
    forecast_result = forecast_result.append(pred)
forecast_result = forecast_result.sort_values(['id', 'ds'])
```

    100%|███████████████████████████████████████████████████████████████████████████████| 20/20 [1:41:04<00:00, 303.20s/it]
    


```python
forecast_result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ds</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-07-01</td>
      <td>834.933588</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2017-07-02</td>
      <td>937.668163</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2017-07-03</td>
      <td>892.186606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2017-07-04</td>
      <td>668.670298</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2017-07-05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>19</td>
      <td>2017-07-27</td>
      <td>497.903196</td>
    </tr>
    <tr>
      <th>27</th>
      <td>19</td>
      <td>2017-07-28</td>
      <td>470.888285</td>
    </tr>
    <tr>
      <th>28</th>
      <td>19</td>
      <td>2017-07-29</td>
      <td>454.039635</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19</td>
      <td>2017-07-30</td>
      <td>469.952823</td>
    </tr>
    <tr>
      <th>30</th>
      <td>19</td>
      <td>2017-07-31</td>
      <td>472.724480</td>
    </tr>
  </tbody>
</table>
<p>620 rows × 3 columns</p>
</div>




```python
forecast_result.to_csv('predict.csv')
```
