import pandas as pd

df = pd.read_csv('datasets_/MSR.csv')
df.head(10)

CVEfixes_df_chunk = pd.read_csv('datasets_/CVEfixes_new.csv', chunksize=1000)
print("CVEfixes_df_chunk is ready")

def read_csv_chunk(chunks):
    df_temp = []
    for chunk in chunks:
        df_temp.append(chunk)
    df = pd.concat(df_temp,ignore_index = True)
    return df

CVEfixes_df = read_csv_chunk(CVEfixes_df_chunk)
CVEfixes_df.head(10)

CVEfixes_df['cve_id'].value_counts()
df['cve_id'].value_counts()
list1 = CVEfixes_df['cve_id'].value_counts().index.tolist()
list1


list2 = df['cve_id'].value_counts().index.tolist()
list2

inter = set.intersection(set(list1), set(list2))
len(inter)

code = df[df['cve_id']=='CVE-2017-12190'].tail(1)['code'].tolist()
print(code[0])



code2 = CVEfixes_df[CVEfixes_df['cve_id']=='CVE-2017-12190'].tail(1)['code'].tolist()
print(code2[0])

CVEfixes_df = CVEfixes_df.drop_duplicates()
