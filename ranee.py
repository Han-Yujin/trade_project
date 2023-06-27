import pandas as pd

# hscode 전처리
hs_export_2017 = pd.read_csv(r'C:\ITWILL\trade_project\data\품목별수출입실적\품목별수출입실적2017.csv', sep='\t' , encoding='utf-16')

hs_export_2017.info()
hs_export_2017['hsCode'] = hs_export_2017['hsCode'].astype('str')
hsCode_str = []
for i in hs_export_2017['hsCode'] :
    if len(i) == 9 :
        i = '0' + i
    else :
        i = i
    hsCode_str.append(i)
hs_export_2017['hsCode'] = hsCode_str


hs_export_2017['HS'] = hs_export_2017['HS'].astype('str')
HS_str = []
for i in hs_export_2017['HS'] :
    if len(i) == 1 :
        i = '0' + i
    else :
        i = i
    HS_str.append(i)
hs_export_2017['HS'] = HS_str


# 국가별 수출입 실적 성장률 top30

# csv 파일 불러오기
country_export_2017 = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별수출입실적\국가별수출입실적2017.csv', sep=',' , encoding='ANSI')
country_export_2018 = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별수출입실적\국가별수출입실적2018.csv', sep=',' , encoding='ANSI')
country_export_2019 = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별수출입실적\국가별수출입실적2019.csv', sep=',' , encoding='ANSI')
country_export_2020 = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별수출입실적\국가별수출입실적2020.csv', sep=',' , encoding='ANSI')
country_export_2021 = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별수출입실적\국가별수출입실적2021.csv', sep=',' , encoding='ANSI')
country_export_2022 = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별수출입실적\국가별수출입실적2022.csv', sep=',' , encoding='ANSI')

'''
# 필요한 칼럼만 추출
country_export_2017 = country_export_2017[['기간','국가명','무역수지']]
country_export_2018 = country_export_2018[['기간','국가명','무역수지']]
country_export_2019 = country_export_2019[['기간','국가명','무역수지']]
country_export_2020 = country_export_2020[['기간','국가명','무역수지']]
country_export_2021 = country_export_2021[['기간','국가명','무역수지']]
country_export_2022 = country_export_2022[['기간','국가명','무역수지']]
'''

# 무역수지 숫자형으로 변형
def convert_to_numeric(df):
    df['수출건수'] = pd.to_numeric(df['수출건수'].str.replace(',', ''), errors='coerce')
    df['수출금액'] = pd.to_numeric(df['수출금액'].str.replace(',', ''), errors='coerce')
    df['수입건수'] = pd.to_numeric(df['수입건수'].str.replace(',', ''), errors='coerce')
    df['수입금액'] = pd.to_numeric(df['수입금액'].str.replace(',', ''), errors='coerce')
    df['무역수지'] = pd.to_numeric(df['무역수지'].str.replace(',', ''), errors='coerce')
    return df

country_export_2017 = convert_to_numeric(country_export_2017)
country_export_2018 = convert_to_numeric(country_export_2018)
country_export_2019 = convert_to_numeric(country_export_2019)
country_export_2020= convert_to_numeric(country_export_2020)
country_export_2021= convert_to_numeric(country_export_2021)
country_export_2022 = convert_to_numeric(country_export_2022)


# 데이터프레임을 리스트로 저장
dfs = [country_export_2017, country_export_2018, country_export_2019, country_export_2020, country_export_2021, country_export_2022]

'''
years = [2017, 2018, 2019, 2020, 2021, 2022]
result_dfs = []
# 데이터프레임별로 연도 정보를 추가하고 새로운 리스트에 저장
for df, year in zip(dfs, years):
    df['연도'] = year
    result_dfs.append(df)
'''

# 무역수지를 국가별로 그룹화하여 연도별 무역수지 합계 계산
trade_balance_sum = pd.concat(dfs).groupby(['국가명', '기간'])['무역수지'].sum()

# 성장률 계산 함수
def calculate_growth_rate(data):
    previous_year = data.shift(1)
    growth_rate = (data - previous_year) / previous_year * 100
    growth_rate = growth_rate.replace([np.inf, -np.inf], np.nan)  # inf(나눈값이 0일때) 값을 NaN으로 처리
    return growth_rate

import numpy as np

# 국가별 성장률 계산
growth_rate = trade_balance_sum.groupby('국가명').apply(calculate_growth_rate)
# 변화량의 평균 퍼센트 계산
average_percentage_change = growth_rate.groupby('국가명').mean()

# 변화량 top30
top_30 = average_percentage_change.nlargest(30).sort_values(ascending=False)
print(top_30)

'''
미령 버진군도       29618.017453
푸에르토리코         4256.685690
세인트 헬레나        3193.740522
몬트세라트          1509.846468
나이지리아          1324.520725
영령 인도양         1177.554113
몬테네그로          1146.519475
중앙아프리카공화국      1078.406147
니우에             904.697181
세인트 키츠 네비스      802.134116
투발루             622.567076
앙귈라             533.537207
우루과이            450.551938
노르웨이            436.379947
토켈라우            395.137144
불가리아            383.056592
대만              344.613795
파라과이            339.496940
지브랄타            332.471251
감비아             307.834890
바레인             278.019873
그리스             190.929517
스와질랜드           187.619858
몰도바             183.597428
안도라             180.447592
뉴질랜드            158.217651
리히텐슈타인          155.227132
가봉              145.988533
캐나다             137.078152
괌               127.145646
'''

# 6년간 무역수지 평군 top30
trade_balance_avg = pd.concat(dfs).groupby('국가명')['무역수지'].mean()
top_30_countries = trade_balance_avg.nlargest(30)
print(top_30_countries)
'''
홍콩        3.357868e+07
베트남       3.043259e+07
중국        2.967481e+07
미국        1.841175e+07
인도        8.986172e+06
필리핀       6.166812e+06
튀르키예      5.103871e+06
폴란드       4.626523e+06
싱가포르      4.561314e+06
멕시코       4.503961e+06
마샬군도      3.874245e+06
헝가리       2.445300e+06
라이베리아     2.109352e+06
슬로바키아     2.082028e+06
태국        1.924086e+06
우즈베키스탄    1.879278e+06
체코공화국     1.679922e+06
벨기에       1.461692e+06
슬로베니아     1.450803e+06
몰타        1.425635e+06
바하마       1.357289e+06
파나마       1.294814e+06
토고        1.129011e+06
방글라데시     1.027389e+06
이집트       9.570855e+05
파키스탄      7.530623e+05
그리스       6.307822e+05
버뮤다       6.017758e+05
영국        5.855113e+05
요르단       5.169565e+05
'''

# 6년간 수출금액 평군 top30
export_balance_avg = pd.concat(dfs).groupby('국가명')['수출금액'].mean()
top_30_countries = export_balance_avg.nlargest(30).sort_values()
print(top_30_countries)
'''
국가명
스페인          2.828526e+06
헝가리          3.131894e+06
벨기에          3.494810e+06
프랑스          3.775377e+06
마샬군도         3.982076e+06
사우디아라비아      4.047978e+06
이탈리아         4.162787e+06
아랍에미리트 연합    4.173726e+06
브라질          4.797295e+06
네덜란드         5.222427e+06
폴란드          5.476394e+06
캐나다          6.007699e+06
영국           6.127655e+06
튀르키예         6.287752e+06
러시아 연방       7.534888e+06
태국           7.956201e+06
인도네시아        8.327677e+06
말레이시아        9.423554e+06
독일           9.549334e+06
필리핀          1.001465e+07
멕시코          1.091722e+07
호주           1.200911e+07
싱가포르         1.339741e+07
인도           1.536145e+07
대만           1.971620e+07
일본           2.858844e+07
홍콩           3.546561e+07
베트남          5.179274e+07
미국           8.240951e+07
중국           1.486192e+08
'''

country = pd.read_csv(r'C:\ITWILL\trade_project\data\국가별 경제현황.csv' , encoding='ANSI')
country = country[['한글 국가명','국내총생산(GDP)','1인당 총생산(GDP)']]
country['국가명'] = country['한글 국가명']


# 국가를 cluster 하여 각 군집을 대표하는 나라 뽑기

import pandas as pd
df1= pd.read_csv(r'C:\ITWILL\trade_project\data\country_2022.csv' , encoding='utf-8')
df1.head()

df2 = pd.read_csv(r'C:\ITWILL\trade_project\data\GDP.csv' , encoding='utf-8')
df2.head()

merged_df = pd.merge(df1, df2, on='국가명')
merged_df.head()
merged_df.to_csv(r'C:\ITWILL\trade_project\data\최종나라목록_2022.csv', encoding='ANSI')

# 필요한 패키지 불러오기
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 원본 데이터를 복사해서 전처리하기 (원본 데이터를 가지고 바로 전처리하지 않는다)
processed_data = merged_df.copy()
processed_data.info()

processed_data['수출건수'] = pd.to_numeric(processed_data['수출건수'].str.replace(',', ''), errors='coerce')
processed_data['수출금액'] = pd.to_numeric(processed_data['수출금액'].str.replace(',', ''), errors='coerce')
processed_data['수입건수'] = pd.to_numeric(processed_data['수입건수'].str.replace(',', ''), errors='coerce')
processed_data['수입금액'] = pd.to_numeric(processed_data['수입금액'].str.replace(',', ''), errors='coerce')
processed_data['무역수지'] = pd.to_numeric(processed_data['무역수지'].str.replace(',', ''), errors='coerce')

processed_data = convert_to_numeric(processed_data)

# 데이터 전처리 - 정규화를 위한 작업
scaler = preprocessing.MinMaxScaler()
processed_data[['수출건수', '수출금액','수입건수','수입금액','무역수지','국내총생산','1인당 총생산']] = scaler.fit_transform(processed_data[['수출건수', '수출금액','수입건수','수입금액','무역수지','국내총생산','1인당 총생산']])

processed_data.head()
processed_data.describe()
processed_data.isnull()
processed_data = processed_data.dropna()

# 화면(figure) 생성
plt.figure(figsize = (10, 6))

# K = 3으로 클러스터링
estimator = KMeans(n_clusters = 3)

# 클러스터링 생성
cluster = estimator.fit_predict(processed_data[['수출건수','수출금액','수입건수','수입금액','무역수지','국내총생산','1인당 총생산']])



# 데이터 축소
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(processed_data[['수출건수', '수출금액', '수입건수', '수입금액', '무역수지', '국내총생산', '1인당 총생산']])

# 군집 결과 시각화
cluster_labels = estimator.labels_

fig, ax = plt.subplots()
scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)

# 국가명 표시
for i, txt in enumerate(processed_data['국가명']):
    ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))

plt.xlabel('PC1')
plt.ylabel('국가명')
plt.title('Clustering Result')
plt.show()





# 새 코드를 짜보아요....

features = processed_data.drop('국가명', axis=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
processed_data['군집'] = kmeans.labels_
processed_data.info()
grouped = processed_data.groupby('군집')
processed_data['군집'].value_counts()