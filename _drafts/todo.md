1. continuous random variable derivations for cdf and expected value, and variance for different distributions.
2. difference between regression in statistics vs in supervised learning
3. maximum likelyhood estimation principle
4. Value of regularization parameter in gradient descent vs normal equation
5. Primality Test https://en.wikipedia.org/wiki/Primality_test
6. Euclid's Algorithm, https://en.wikipedia.org/wiki/Euclidean_algorithm, https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm, http://e-maxx.ru/algo/extended_euclid_algorithm#3
7. Legal Side of Open Source https://opensource.guide/legal/#why-do-people-care-so-much-about-the-legal-side-of-open-source
8. Chinese Remainder Theorem
9. Unsupervised learning
10. SVM
11. AdaBoost
12. Deaggregation of time series data
13. http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html

# total
df_tmp = w.df_sms_test[:200]
len(df_tmp)

# not none
df_tmp = df_tmp[df_tmp['PATIENT_NAME'] != 'None']
len(df_tmp)

# not single character names
def name_length(name):
    return min([len(_) for _ in name.split()])

df_tmp['LENGTH'] = df_tmp.PATIENT_NAME.apply(name_length)
df_tmp = df_tmp[df_tmp.LENGTH > 1]
len(df_tmp)

# change from None
count = 0
for idx, row in tqdm(df_tmp.iterrows(), total = len(df_tmp)):
    if row['URN'].read() == 'None' and 'KH' in row['URN_LEV_1']:
        count +=1
count

# change from Previous
df_tmp['CHANGE'] = [0] * len(df_tmp)
for idx, row in tqdm(df_tmp.iterrows(), total = len(df_tmp)):
    count = 0
    old_urn = row['URN'].read()
    new_urn = row['URN_LEV_1']
    if old_urn != 'None' and new_urn != '':
        if set(old_urn.split(',')) != set(new_urn.split(',')):
            count += 1
    df_tmp.at[idx, 'CHANGE'] = count



from tqdm import tqdm
from utility import lev_dist
import pandas as pd

def name_length(name):
    return min([len(_) for _ in name.split()])

date_filter = pd.Timestamp('2018-02-01')
w.df_sms_test = w.df_sms_test[w.df_sms_test.SMS_DATE > date_filter]
w.df_sms_test = w.df_sms_test[w.df_sms_test.PATIENT_NAME != 'None']
w.df_sms_test = w.df_sms_test[~pd.isnull(w.df_sms_test.PATIENT_NAME)]
w.df_sms_test['LENGTH'] = w.df_sms_test.PATIENT_NAME.apply(name_length)
w.df_sms_test = w.df_sms_test[w.df_sms_test.LENGTH > 3]
w.df_sms_test['URN_LEV_1'] = [''] * len(w.df_sms_test)

for id_sms, row_sms in tqdm(w.df_sms_test[:200].iterrows(), total=len(w.df_sms_test[:200])):
    names = row_sms['PATIENT_NAME'].split()
    sms_date = row_sms['SMS_DATE']
    candidate_ids = []
    for id_pt, row_pt in tqdm(w.df_patient.iterrows(), total=len(w.df_patient)):
        urn = row_pt['PATIENT_ID']
        if urn in w.latest_encounter_dict:
            pt_names = [row_pt['FIRST_NAME'], row_pt['FAMILY_NAME']]
            match_score = 0
            for x in pt_names:
                if x is None:
                    continue
                for y in names:
                    if y is None:
                        continue
                    if lev_dist(x, y) <= 1:
                        match_score += 1
            if match_score >= 2:
                lower_limit = sms_date - pd.DateOffset(days=1)
                upper_limit = sms_date + pd.DateOffset(months=4)
                if w.latest_encounter_dict[urn] > lower_limit and w.latest_encounter_dict[urn] < upper_limit:
                    candidate_ids.append(urn)
    w.df_sms_test.at[id_sms, 'URN_LEV_1'] = ",".join(list(set(candidate_ids)))