import pandas as pd


columns_of_interest = ['age', 'MCV_fL', 'PT_percent', 'LDH_UI_L',
                       'MCHC_g_L', 'WBC_G_L', 'Fibrinogen_g_L', 'Monocytes_G_L',
                       'Platelets_G_L', 'Lymphocytes_G_L'
                       ]
master_df = pd.read_csv('./all_origins.csv')

bad_performers = ['lagos', 'kolkata', 'milano', 'dallas']
reference1 = master_df[master_df.origin == 'maastricht']
reference2 = master_df[master_df.origin == 'wroclaw']

for c in columns_of_interest:
    print()
    print(c)
    print(f'maastricht: {reference1[c].astype(float).mean()}, {reference1[c].astype(float).std()}')
    print(f'wroclaw: {reference2[c].astype(float).mean()}, {reference2[c].astype(float).std()}')
    print()
    for b in bad_performers:
        df = master_df[master_df.origin == b]
        print(f'{b}: {df[c].astype(float).mean()}, {df[c].astype(float).std()}')