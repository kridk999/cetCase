import pandas as pd


df = pd.read_csv (r'assets\data.csv')


df['date_time'] = pd.to_datetime(df['date_time'],utc=True).dt.tz_convert('CET')
df = df.set_index('date_time')

df["Hour"] = df.index.tz_convert("Europe/Copenhagen").hour + 1

df.info()

df.to_csv(r'assets\processed_data.csv')

print(df)

