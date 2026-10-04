import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

# ------------------------------------------------------------------
# 1. LOAD + EXPLORE
# ------------------------------------------------------------------
df_data = pd.read_csv('nobel_prize_data.csv')

print('Shape:', df_data.shape)
print(df_data.columns.tolist())
print(df_data.head())
print('First year:', df_data.year.min(), '| Last year:', df_data.year.max())

# ------------------------------------------------------------------
# 2. DUPLICATES + NaN
# ------------------------------------------------------------------
print(f'Any duplicates? {df_data.duplicated().values.any()}')
print(f'Any NaN values? {df_data.isna().values.any()}')
print(df_data.isna().sum())

col_subset = ['year', 'category', 'laureate_type', 'birth_date', 'full_name', 'organization_name']
print(df_data.loc[df_data.birth_date.isna()][col_subset])

col_subset = ['year', 'category', 'laureate_type', 'full_name', 'organization_name']
print(df_data.loc[df_data.organization_name.isna()][col_subset])

# ------------------------------------------------------------------
# 3. TYPE CONVERSIONS
# ------------------------------------------------------------------
df_data.birth_date = pd.to_datetime(df_data.birth_date)

separated_values = df_data.prize_share.str.split('/', expand=True)
numerator = pd.to_numeric(separated_values[0])
denominator = pd.to_numeric(separated_values[1])
df_data['share_pct'] = numerator / denominator

df_data.info()
print(df_data[['prize_share', 'share_pct']].head())

# ------------------------------------------------------------------
# 4. MEN vs WOMEN DONUT
# ------------------------------------------------------------------
sex_counts = df_data.sex.value_counts()
fig = px.pie(labels=sex_counts.index,
             values=sex_counts.values,
             title='Percentage of Male vs. Female Winners',
             names=sex_counts.index,
             hole=0.4)
fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')
fig.show()

# First 3 women
print(df_data[df_data.sex == 'Female'].sort_values('year', ascending=True)[:3])

# Repeat winners
is_winner = df_data.duplicated(subset=['full_name'], keep=False)
multiple_winners = df_data[is_winner]
print(f'There are {multiple_winners.full_name.nunique()} winners who were awarded the prize more than once.')
print(multiple_winners[['year', 'category', 'laureate_type', 'full_name']])

# ------------------------------------------------------------------
# 5. PRIZES PER CATEGORY
# ------------------------------------------------------------------
print('Number of categories:', df_data.category.nunique())

prizes_per_category = df_data.category.value_counts()
v_bar = px.bar(x=prizes_per_category.index,
               y=prizes_per_category.values,
               color=prizes_per_category.values,
               color_continuous_scale='Aggrnyl',
               title='Number of Prizes Awarded per Category')
v_bar.update_layout(xaxis_title='Nobel Prize Category',
                    coloraxis_showscale=False,
                    yaxis_title='Number of Prizes')
v_bar.show()

# Economics prize start
print(df_data[df_data.category == 'Economics'].sort_values('year')[:3])

# Split by sex
cat_men_women = df_data.groupby(['category', 'sex'], as_index=False).agg({'prize': 'count'})
cat_men_women.sort_values('prize', ascending=False, inplace=True)

v_bar_split = px.bar(cat_men_women, x='category', y='prize', color='sex',
                     title='Number of Prizes Awarded per Category split by Men and Women')
v_bar_split.update_layout(xaxis_title='Nobel Prize Category', yaxis_title='Number of Prizes')
v_bar_split.show()

# ------------------------------------------------------------------
# 6. PRIZES OVER TIME + ROLLING AVERAGE
# ------------------------------------------------------------------
prize_per_year = df_data.groupby(by='year').count().prize
moving_average = prize_per_year.rolling(window=5).mean()

yearly_avg_share = df_data.groupby(by='year').agg({'share_pct': 'mean'})
share_moving_average = yearly_avg_share.rolling(window=5).mean()

plt.figure(figsize=(16, 8), dpi=100)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(ticks=np.arange(1900, 2021, step=5), fontsize=14, rotation=45)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlim(1900, 2020)
ax2.invert_yaxis()  # inverted: lower share = more people per prize

ax1.scatter(x=prize_per_year.index, y=prize_per_year.values,
            c='dodgerblue', alpha=0.7, s=100)
ax1.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=3)
ax2.plot(prize_per_year.index, share_moving_average.values, c='grey', linewidth=3)

ax1.set_ylabel('Number of prizes', fontsize=14)
ax2.set_ylabel('Avg prize share (inverted)', fontsize=14)
plt.show()

# ------------------------------------------------------------------
# 7. TOP 20 COUNTRIES + CHOROPLETH
# ------------------------------------------------------------------
top_countries = df_data.groupby(['birth_country_current'], as_index=False).agg({'prize': 'count'})
top_countries.sort_values(by='prize', inplace=True)
top20_countries = top_countries[-20:]

h_bar = px.bar(x=top20_countries.prize,
               y=top20_countries.birth_country_current,
               orientation='h',
               color=top20_countries.prize,
               color_continuous_scale='Viridis',
               title='Top 20 Countries by Number of Prizes')
h_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Country',
                    coloraxis_showscale=False)
h_bar.show()

df_countries = df_data.groupby(['birth_country_current', 'ISO'], as_index=False).agg({'prize': 'count'})
df_countries = df_countries.sort_values('prize', ascending=False)

world_map = px.choropleth(df_countries,
                          locations='ISO',
                          color='prize',
                          hover_name='birth_country_current',
                          color_continuous_scale=px.colors.sequential.matter)
world_map.update_layout(coloraxis_showscale=True)
world_map.show()

# Category breakdown by country
cat_country = df_data.groupby(['birth_country_current', 'category'], as_index=False).agg({'prize': 'count'})
cat_country.sort_values(by='prize', ascending=False, inplace=True)

merged_df = pd.merge(cat_country, top20_countries, on='birth_country_current')
merged_df.columns = ['birth_country_current', 'category', 'cat_prize', 'total_prize']
merged_df.sort_values(by='total_prize', inplace=True)

cat_cntry_bar = px.bar(x=merged_df.cat_prize,
                       y=merged_df.birth_country_current,
                       color=merged_df.category,
                       orientation='h',
                       title='Top 20 Countries by Number of Prizes and Category')
cat_cntry_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Country')
cat_cntry_bar.show()

# Prizes by country over time (top 10 only, for readability)
prize_by_year = df_data.groupby(by=['birth_country_current', 'year'], as_index=False).count()
prize_by_year = prize_by_year.sort_values('year')[['year', 'birth_country_current', 'prize']]

cumulative_prizes = (prize_by_year.groupby(by=['birth_country_current', 'year']).sum()
                     .groupby(level=[0]).cumsum())
cumulative_prizes.reset_index(inplace=True)

top10 = top20_countries.tail(10).birth_country_current
l_chart = px.line(cumulative_prizes[cumulative_prizes.birth_country_current.isin(top10)],
                  x='year', y='prize',
                  color='birth_country_current',
                  hover_name='birth_country_current')
l_chart.update_layout(xaxis_title='Year', yaxis_title='Number of Prizes')
l_chart.show()

# ------------------------------------------------------------------
# 8. RESEARCH ORGANISATIONS, CITIES, BIRTH CITIES
# ------------------------------------------------------------------
top20_orgs = df_data.organization_name.value_counts()[:20]
top20_orgs.sort_values(ascending=True, inplace=True)
org_bar = px.bar(x=top20_orgs.values, y=top20_orgs.index, orientation='h',
                 color=top20_orgs.values,
                 color_continuous_scale=px.colors.sequential.haline,
                 title='Top 20 Research Institutions by Number of Prizes')
org_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='Institution',
                      coloraxis_showscale=False)
org_bar.show()

print(df_data.organization_name.value_counts()[['Harvard University', 'University of Chicago']])

top20_org_cities = df_data.organization_city.value_counts()[:20]
top20_org_cities.sort_values(ascending=True, inplace=True)
city_bar2 = px.bar(x=top20_org_cities.values, y=top20_org_cities.index, orientation='h',
                   color=top20_org_cities.values,
                   color_continuous_scale=px.colors.sequential.Plasma,
                   title='Which Cities Do the Most Research?')
city_bar2.update_layout(xaxis_title='Number of Prizes', yaxis_title='City',
                        coloraxis_showscale=False)
city_bar2.show()

top20_cities = df_data.birth_city.value_counts()[:20]
top20_cities.sort_values(ascending=True, inplace=True)
city_bar = px.bar(x=top20_cities.values, y=top20_cities.index, orientation='h',
                  color=top20_cities.values,
                  color_continuous_scale=px.colors.sequential.Plasma,
                  title='Where were the Nobel Laureates Born?')
city_bar.update_layout(xaxis_title='Number of Prizes', yaxis_title='City of Birth',
                       coloraxis_showscale=False)
city_bar.show()

# % of US prizes from New York
us_total = (df_data.birth_country_current == 'United States').sum()
nyc = df_data.birth_city.str.contains('New York', na=False).sum()
print(f'US prizes from New York: {nyc / us_total:.1%}')
print(df_data.birth_city.value_counts()[['London', 'Paris', 'Vienna']])

# ------------------------------------------------------------------
# 9. SUNBURST
# ------------------------------------------------------------------
country_city_org = df_data.groupby(by=['organization_country',
                                       'organization_city',
                                       'organization_name'], as_index=False).agg({'prize': 'count'})
country_city_org = country_city_org.sort_values('prize', ascending=False)

burst = px.sunburst(country_city_org,
                    path=['organization_country', 'organization_city', 'organization_name'],
                    values='prize',
                    title='Where do Discoveries Take Place?')
burst.show()

# ------------------------------------------------------------------
# 10. WINNING AGE
# ------------------------------------------------------------------
birth_years = df_data.birth_date.dt.year
df_data['winning_age'] = df_data.year - birth_years

cols = ['full_name', 'category', 'year', 'winning_age', 'motivation']
print('Oldest:\n', df_data.nlargest(n=1, columns='winning_age')[cols])
print('Youngest:\n', df_data.nsmallest(n=1, columns='winning_age')[cols])
print(df_data.winning_age.describe())

# Histograms with different bin sizes
for b in [10, 20, 30, 50]:
    plt.figure(figsize=(8, 4), dpi=100)
    sns.histplot(data=df_data, x='winning_age', bins=b)
    plt.xlabel('Age')
    plt.title(f'Distribution of Age on Receipt of Prize (bins={b})')
    plt.show()

# Age over time (all categories)
plt.figure(figsize=(8, 4), dpi=100)
with sns.axes_style('whitegrid'):
    sns.regplot(data=df_data, x='year', y='winning_age',
                lowess=True,
                scatter_kws={'alpha': 0.4},
                line_kws={'color': 'black'})
plt.show()

# Box plot by category
plt.figure(figsize=(8, 4), dpi=100)
with sns.axes_style('whitegrid'):
    sns.boxplot(data=df_data, x='category', y='winning_age')
plt.show()

# lmplot: separate chart per category
with sns.axes_style('whitegrid'):
    sns.lmplot(data=df_data, x='year', y='winning_age',
               row='category', lowess=True, aspect=2,
               scatter_kws={'alpha': 0.6},
               line_kws={'color': 'black'})
plt.show()

# lmplot: all categories on one chart
with sns.axes_style('whitegrid'):
    sns.lmplot(data=df_data, x='year', y='winning_age',
               hue='category', lowess=True, aspect=2,
               scatter_kws={'alpha': 0.5},
               line_kws={'linewidth': 5})
plt.show()