import pandas as pd
data = pd.read_csv('fortune_2000_in_2021.csv')

# Editing data using pandas
def clean_number(a):
    new_a=''
    for i in a:
        if i=='$' or i==',':
            pass
        elif i=='B':
            new_a = float(new_a)*1
        elif i=='M':
            new_a = float(new_a)*1/1000
        else:
            new_a+= i
    return new_a

def clean(row):
    row['Profit'] = clean_number(row.Profit)
    row['Sales'] = clean_number(row.Sales)
    row['Assets'] = clean_number(row.Assets)
    row['Market_value'] = clean_number(row.Market_value)
    return row

data = data.apply(clean,axis='columns')
data.to_csv('new_data.csv',index=False)