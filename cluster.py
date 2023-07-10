from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 국가를 cluster 하여 각 군집을 대표하는 나라 뽑기
df1= pd.read_csv(r'C:\ITWILL\trade_project\data\country_2022.csv' , encoding='utf-8')
df1.head()

df2 = pd.read_csv(r'C:\ITWILL\trade_project\data\GDP.csv' , encoding='utf-8')
df2.head()

df3 = pd.merge(df1, df2, on='국가명')
df3.head()
df3.to_csv(r'C:\ITWILL\trade_project\data\최종나라목록_2022.csv', encoding='ANSI', index=False)

# 군집분석 용 변수 추가
df4 = pd.read_csv(r'C:\ITWILL\trade_project\data\환율.csv', encoding='ANSI')
df5 = pd.read_csv(r'C:\ITWILL\trade_project\data\국제금리.csv', encoding='ANSI')
df4.info()
df5.info()
country_name = []
for i in df4['국가(통화단위)별'] :
    a = i.split('(')
    country_name.append(a[0])

df4['국가명'] = country_name
df4['환율'] = df4['2021']
df4 = df4[['국가명','환율']]
df5['국가명'] = df5['국가별']
df5['대출금리'] = df5['2021.2']
df5 = df5[['대출금리','국가명']]


len(df3) # 144
merged = pd.merge(df3,df4, on='국가명')
len(merged) # 108
merged2 = pd.merge(merged,df5, on='국가명')
merged2 = merged2[['국가명','무역수지','국내총생산','1인당 총생산','환율','대출금리']] 
len(merged2) # 78, data개수손실이 커 대출금리는 사용X

# 결측치 처리
merged.isnull().sum()
merged = merged.dropna()

merged.info()

merged.corr(method='pearson')
merged.describe()

# 각 변수 float형으로 변환

def re_float(a) :
    a = a.str.replace(',', '').astype('float')
    return a

merged['수출건수'] = re_float(merged['수출건수'])
merged['수출금액'] = re_float(merged['수출금액'])
merged['수입건수'] = re_float(merged['수입건수'])
merged['수입금액'] = re_float(merged['수입금액'])
merged['무역수지'] = re_float(merged['무역수지'])


# 자료 표준화
merged2 = merged.drop(['국가명'], axis=1)
scaler = StandardScaler()
merged2 = scaler.fit_transform(merged2)


# best cluster 찾기
size = range(1, 11) # k값 범위
inertia = [] # 응집도 (중심점과 포인트 간 거리 제곱합)

for k in size : 
    obj = KMeans(n_clusters = k) 
    model = obj.fit(merged2)
    inertia.append(model.inertia_) 

print(inertia)


# 시각화
plt.plot(size, inertia, '-o')
plt.xticks(size)
plt.show()

# 클러스터링
kmeans = KMeans(n_clusters=5, random_state=10)
clusters = kmeans.fit(merged2)

merged['cluster'] = clusters.labels_
merged['cluster'].value_counts()

merged.groupby('cluster').mean()
merged.to_csv(r'C:\ITWILL\trade_project\data\merged.csv', encoding='ANSI')


# 각 클러스터 상위 25% 추출
cluster1 = merged[merged['cluster']==0]
cluster1 = cluster1[cluster1['무역수지'] >= cluster1['무역수지'].quantile(.75)]
cluster1.info()
cluster1.to_csv(r'C:\ITWILL\trade_project\data\cluster0.csv', encoding='ANSI')

cluster2 = merged[merged['cluster']==2]
cluster2 = cluster2[cluster2['무역수지'] >= cluster2['무역수지'].quantile(.75)]
cluster2.info()
cluster2.to_csv(r'C:\ITWILL\trade_project\data\cluster2.csv', encoding='ANSI')
