# Table of contents


* [Imports, data reading and descriptive analysis](#imp)
    * [Missing values imputation](#miss)

* [Best model selection (TS CV)](#best)
    * ["Complete" time series](#comp)
    * ["Short" time series](#short)

* [Forecasting and conclusions](#forecast)

## Imports, data reading and descriptive analysis <a class="anchor" id="imp"></a>


```python
import numpy as np
import pandas as pd

import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
```


```python
from ts_cv import ts_cv_metrics
from wrappers import WrapperAutoArima, WrapperProphet, WrapperHoltWinters
```


```python
df = pd.read_csv('train.csv')
df['dt'] = pd.to_datetime(df['dt'])
df.columns = ['id', 'ds', 'y']
df_agg = df.groupby('id').agg({'ds': ['min', 'max', 'count'], 'y': ['mean', 'std']}).reset_index()
df_agg
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>id</th>
      <th colspan="3" halign="left">ds</th>
      <th colspan="2" halign="left">y</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>453</td>
      <td>776.880353</td>
      <td>192.248316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>453</td>
      <td>485.492274</td>
      <td>122.866411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>448</td>
      <td>595.209598</td>
      <td>108.123227</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>448</td>
      <td>435.170982</td>
      <td>148.888847</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>385.991156</td>
      <td>54.456945</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>454.067849</td>
      <td>109.136251</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>640.762749</td>
      <td>166.423599</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>450</td>
      <td>634.192000</td>
      <td>115.375308</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>1549.520748</td>
      <td>297.530101</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>779.500443</td>
      <td>184.243248</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>675.758163</td>
      <td>104.065112</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>452</td>
      <td>513.794027</td>
      <td>135.294493</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>450</td>
      <td>776.747111</td>
      <td>283.941149</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>687.877211</td>
      <td>128.385238</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>450</td>
      <td>606.364222</td>
      <td>177.961598</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>453</td>
      <td>807.902428</td>
      <td>180.486750</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>451</td>
      <td>446.958980</td>
      <td>97.033838</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>294</td>
      <td>484.503741</td>
      <td>147.873661</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>2016-01-02</td>
      <td>2017-06-30</td>
      <td>448</td>
      <td>646.840625</td>
      <td>177.672308</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
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
# number of days of week (0 - Sunday, 1 - Monday, etc.) for each store
df.pivot_table(values='y', index='id', columns='day',aggfunc='count').reset_index()
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
      <td>1</td>
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
      <td>2</td>
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
      <td>3</td>
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
      <td>4</td>
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
      <td>5</td>
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
      <td>6</td>
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
      <td>7</td>
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
      <td>8</td>
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
      <td>9</td>
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
      <td>10</td>
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
      <td>11</td>
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
      <td>12</td>
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
      <td>13</td>
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
      <td>14</td>
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
      <td>15</td>
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
      <td>16</td>
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
      <td>17</td>
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
      <td>18</td>
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
      <td>19</td>
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
pivot_df.groupby('mnth').min().reset_index()
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
      <th>mnth</th>
      <th>id</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
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
      <th>1</th>
      <td>2016-02-01</td>
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
      <th>2</th>
      <td>2016-03-01</td>
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
      <th>3</th>
      <td>2016-04-01</td>
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
      <th>4</th>
      <td>2016-05-01</td>
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
      <th>5</th>
      <td>2016-06-01</td>
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
      <th>6</th>
      <td>2016-07-01</td>
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
      <th>7</th>
      <td>2016-08-01</td>
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
      <th>8</th>
      <td>2016-09-01</td>
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
      <th>9</th>
      <td>2016-10-01</td>
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
      <th>10</th>
      <td>2016-11-01</td>
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
      <th>11</th>
      <td>2016-12-01</td>
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
      <th>12</th>
      <td>2017-01-01</td>
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
      <th>13</th>
      <td>2017-02-01</td>
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
      <th>14</th>
      <td>2017-03-01</td>
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
      <th>15</th>
      <td>2017-04-01</td>
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
      <th>16</th>
      <td>2017-05-01</td>
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
      <th>17</th>
      <td>2017-06-01</td>
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
df = df[df['ds'] >= pd.to_datetime('2016-03-01')]

dates_list = pd.date_range(df['ds'].min(), df['ds'].max())
dates_list = dates_list[dates_list.weekday != 2]  # exclude second day of week (day off)

tmp_dict = pd.DataFrame(dates_list).copy()
tmp_dict.columns = ['ds']
for ts in range(len(np.unique(df['id']))):
    tmp_dict['id'] = ts
    df = pd.merge(df, tmp_dict, how='outer')

df = df.sort_values(['id', 'ds'])

df['day'] = df['ds'].dt.dayofweek  # day of week
df['mnth'] = df['ds'].apply(lambda x: x.replace(day=1))  # month
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
short_ts = df_agg[df_agg['ds']['count']<400].index.tolist()
complete_ts = df_agg[df_agg['ds']['count']>=400].index.tolist()
```


```python
# some plots

plt.figure(figsize=(12, 6))
for ts in short_ts:
    sns.lineplot(data=df.loc[df['id']==ts], x='ds', y='y', palette='Blues')
plt.title('"Short" Time Series')
```




    Text(0.5, 1.0, '"Short" Time Series')




    
![png](output_10_1.png)
    



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
def ts_metrics_all(ts_list, models_list, df, cv_folds=3, horizon=31):
    result = []
    for ts in tqdm(ts_list):    
        tmp = df.loc[(df['id']==ts)]
        for model in models_list:
            mean_smae, mean_mape, model_name = ts_cv_metrics(tmp, model, cv_folds=cv_folds, horizon=horizon)
            res = {'id': ts, 'model': model_name, 'smae': mean_smae, 'mape': mean_mape}
            result.append(res)
    return result
```


```python
hw_model = WrapperHoltWinters()
arima_model = WrapperAutoArima()
prophet_model = WrapperProphet()
models = [hw_model, arima_model, prophet_model]
```


```python
metrics_complete_ts = ts_metrics_all(complete_ts, models, df, cv_folds=5, horizon=31)
metrics_complete_ts = pd.DataFrame(metrics_complete_ts)
metrics_complete_ts
```

    100%|███████████████████████████████████████████████████████████████████████████████| 15/15 [1:25:55<00:00, 343.71s/it]
    




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
      <td>0.195047</td>
      <td>0.199159</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>AutoArima</td>
      <td>0.178614</td>
      <td>0.191724</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Prophet</td>
      <td>0.120531</td>
      <td>0.123942</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>HoltWinters</td>
      <td>0.188494</td>
      <td>0.185614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>AutoArima</td>
      <td>0.163679</td>
      <td>0.156482</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>Prophet</td>
      <td>0.134649</td>
      <td>0.136642</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>HoltWinters</td>
      <td>0.167762</td>
      <td>0.162852</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>AutoArima</td>
      <td>0.136447</td>
      <td>0.134170</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>Prophet</td>
      <td>0.100451</td>
      <td>0.100364</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>HoltWinters</td>
      <td>0.314029</td>
      <td>0.328993</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>AutoArima</td>
      <td>0.242179</td>
      <td>0.269331</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>Prophet</td>
      <td>0.150955</td>
      <td>0.159472</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>HoltWinters</td>
      <td>0.233058</td>
      <td>0.241841</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>AutoArima</td>
      <td>0.187230</td>
      <td>0.194210</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>Prophet</td>
      <td>0.150049</td>
      <td>0.163757</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6</td>
      <td>HoltWinters</td>
      <td>0.227989</td>
      <td>0.228961</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>AutoArima</td>
      <td>0.190754</td>
      <td>0.196153</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>Prophet</td>
      <td>0.157743</td>
      <td>0.174384</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>HoltWinters</td>
      <td>0.186199</td>
      <td>0.177341</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7</td>
      <td>AutoArima</td>
      <td>0.132788</td>
      <td>0.131292</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>Prophet</td>
      <td>0.095446</td>
      <td>0.096572</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9</td>
      <td>HoltWinters</td>
      <td>0.242500</td>
      <td>0.235755</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9</td>
      <td>AutoArima</td>
      <td>0.171521</td>
      <td>0.177408</td>
    </tr>
    <tr>
      <th>23</th>
      <td>9</td>
      <td>Prophet</td>
      <td>0.130478</td>
      <td>0.141234</td>
    </tr>
    <tr>
      <th>24</th>
      <td>11</td>
      <td>HoltWinters</td>
      <td>0.215427</td>
      <td>0.215640</td>
    </tr>
    <tr>
      <th>25</th>
      <td>11</td>
      <td>AutoArima</td>
      <td>0.200275</td>
      <td>0.200806</td>
    </tr>
    <tr>
      <th>26</th>
      <td>11</td>
      <td>Prophet</td>
      <td>0.140403</td>
      <td>0.150781</td>
    </tr>
    <tr>
      <th>27</th>
      <td>12</td>
      <td>HoltWinters</td>
      <td>0.248980</td>
      <td>0.255750</td>
    </tr>
    <tr>
      <th>28</th>
      <td>12</td>
      <td>AutoArima</td>
      <td>0.261737</td>
      <td>0.301395</td>
    </tr>
    <tr>
      <th>29</th>
      <td>12</td>
      <td>Prophet</td>
      <td>0.159377</td>
      <td>0.167154</td>
    </tr>
    <tr>
      <th>30</th>
      <td>14</td>
      <td>HoltWinters</td>
      <td>0.202548</td>
      <td>0.196056</td>
    </tr>
    <tr>
      <th>31</th>
      <td>14</td>
      <td>AutoArima</td>
      <td>0.226541</td>
      <td>0.237733</td>
    </tr>
    <tr>
      <th>32</th>
      <td>14</td>
      <td>Prophet</td>
      <td>0.129704</td>
      <td>0.132997</td>
    </tr>
    <tr>
      <th>33</th>
      <td>15</td>
      <td>HoltWinters</td>
      <td>0.210965</td>
      <td>0.211920</td>
    </tr>
    <tr>
      <th>34</th>
      <td>15</td>
      <td>AutoArima</td>
      <td>0.159229</td>
      <td>0.159316</td>
    </tr>
    <tr>
      <th>35</th>
      <td>15</td>
      <td>Prophet</td>
      <td>0.120056</td>
      <td>0.118725</td>
    </tr>
    <tr>
      <th>36</th>
      <td>16</td>
      <td>HoltWinters</td>
      <td>0.203399</td>
      <td>0.200583</td>
    </tr>
    <tr>
      <th>37</th>
      <td>16</td>
      <td>AutoArima</td>
      <td>0.154907</td>
      <td>0.151604</td>
    </tr>
    <tr>
      <th>38</th>
      <td>16</td>
      <td>Prophet</td>
      <td>0.126908</td>
      <td>0.129348</td>
    </tr>
    <tr>
      <th>39</th>
      <td>18</td>
      <td>HoltWinters</td>
      <td>0.213882</td>
      <td>0.211394</td>
    </tr>
    <tr>
      <th>40</th>
      <td>18</td>
      <td>AutoArima</td>
      <td>0.198957</td>
      <td>0.191751</td>
    </tr>
    <tr>
      <th>41</th>
      <td>18</td>
      <td>Prophet</td>
      <td>0.133445</td>
      <td>0.138143</td>
    </tr>
    <tr>
      <th>42</th>
      <td>19</td>
      <td>HoltWinters</td>
      <td>0.238784</td>
      <td>0.236639</td>
    </tr>
    <tr>
      <th>43</th>
      <td>19</td>
      <td>AutoArima</td>
      <td>0.214069</td>
      <td>0.212975</td>
    </tr>
    <tr>
      <th>44</th>
      <td>19</td>
      <td>Prophet</td>
      <td>0.152065</td>
      <td>0.161517</td>
    </tr>
  </tbody>
</table>
</div>



### "Short" time series<a class="anchor" id="short"></a>

For incomplete time series the same approach as for the rest of data was used. The initial idea is that we can forecast using as a train sample only short part of a dataset, which is the closest to the forecasted period of time. Two fold cross-validation was used due to the length of the time series.


```python
metrics_short_ts = ts_metrics_all(short_ts, models, df[df['ds']>=pd.to_datetime('2017-01-02')], cv_folds=2, horizon=31)
metrics_short_ts = pd.DataFrame(metrics_short_ts)
metrics_short_ts
```

    100%|███████████████████████████████████████████████████████████████████████████████████| 5/5 [16:19<00:00, 195.82s/it]
    




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
      <td>0.205914</td>
      <td>0.210225</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>AutoArima</td>
      <td>0.179538</td>
      <td>0.197524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Prophet</td>
      <td>0.158933</td>
      <td>0.173351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>HoltWinters</td>
      <td>0.163386</td>
      <td>0.155376</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>AutoArima</td>
      <td>0.143364</td>
      <td>0.140614</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>Prophet</td>
      <td>0.114609</td>
      <td>0.116827</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>HoltWinters</td>
      <td>0.183850</td>
      <td>0.182685</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>AutoArima</td>
      <td>0.112023</td>
      <td>0.110992</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>Prophet</td>
      <td>0.099079</td>
      <td>0.103406</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13</td>
      <td>HoltWinters</td>
      <td>0.126078</td>
      <td>0.115859</td>
    </tr>
    <tr>
      <th>10</th>
      <td>13</td>
      <td>AutoArima</td>
      <td>0.128237</td>
      <td>0.123426</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>Prophet</td>
      <td>0.090895</td>
      <td>0.093208</td>
    </tr>
    <tr>
      <th>12</th>
      <td>17</td>
      <td>HoltWinters</td>
      <td>0.239347</td>
      <td>0.249304</td>
    </tr>
    <tr>
      <th>13</th>
      <td>17</td>
      <td>AutoArima</td>
      <td>0.201788</td>
      <td>0.195898</td>
    </tr>
    <tr>
      <th>14</th>
      <td>17</td>
      <td>Prophet</td>
      <td>0.136196</td>
      <td>0.137200</td>
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
for ts in tqdm(np.unique(df['id'])):
    if ts in short_ts:
        tmp = df.loc[(df['id']==ts)&(df['ds'] >= pd.to_datetime('2017-01-02'))]
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

    100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [33:07<00:00, 99.39s/it]
    


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
      <td>495.439183</td>
    </tr>
    <tr>
      <th>27</th>
      <td>19</td>
      <td>2017-07-28</td>
      <td>468.554276</td>
    </tr>
    <tr>
      <th>28</th>
      <td>19</td>
      <td>2017-07-29</td>
      <td>451.819106</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19</td>
      <td>2017-07-30</td>
      <td>467.826323</td>
    </tr>
    <tr>
      <th>30</th>
      <td>19</td>
      <td>2017-07-31</td>
      <td>470.567212</td>
    </tr>
  </tbody>
</table>
<p>620 rows × 3 columns</p>
</div>




```python
forecast_result.to_csv('predict.csv')
```
