import pandas as pd
import random
from datetime import timedelta
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import pickle as pkl


class SGenerator:
    def __init__(self, df=None, is_laund=False):
        if df != None:
            self.from_banks = df['From Bank'].unique()
            self.to_banks = df['To Bank'].unique()
            self.banks_rate = df[['From Bank', 'To Bank']].value_counts(normalize=True, dropna=False)
            self.banks = pd.DataFrame({'banks':self.banks_rate.index, "prob":self.banks_rate.values})

            self.acc1 = df['Account'].unique()
            self.acc2 = df['Account.1'].unique()
            self.accs_rate = df[['Account', 'Account.1']].value_counts(normalize=True, dropna=False)
            self.accs = pd.DataFrame({'accs':self.accs_rate.index, "prob":self.accs_rate.values})

            self.formats_rate = df[['Payment Format']].value_counts(normalize=True, dropna=False)
            self.formats = pd.DataFrame({'accs':self.formats_rate.index, "prob":self.formats_rate.values})

            self.currs_rate = df[['Receiving Currency']].value_counts(normalize=True, dropna=False)
            self.currs = pd.DataFrame({'accs':self.currs_rate.index, "prob":self.currs_rate.values})

            std_dev_price = df['Amount Received'].std()


            mean_price = df['Amount Received'].mean()
            threshold = 0.5

            df['z_score'] = (df['Amount Received'] - mean_price) / std_dev_price

            filtered_df = df[abs(df['z_score']) <= threshold]

            filtered_df = filtered_df.drop(columns=['z_score'])

            self.amount = filtered_df ['Amount Received']
            self.min_sum = min(self.amount)
            self.max_sum = max(self.amount)
            self.rng = self.max_sum - self.min_sum
            self.n = 100
            self.lrng = self.rng / self.n

            chances = []
            sections = []    

            for i in range(100):
                a = self.min_sum + i * self.lrng
                b = self.min_sum + (i+1) * self.lrng
                sections.append([a, b])

                count = len(df[(df['Amount Received'] > a) & (df['Amount Received'] < b)].index)
                chance = count / len(df.index)
                chances.append(chance)
            
            self.chances = chances
            self.sections = sections
        else:
            with open("src/pickles/sgen_settings.pkl", "rb") as f:
                sett = pkl.load(f)
            
            self.from_banks = sett['from_banks']
            self.to_banks = sett['to_banks']
            self.banks_rate = sett['banks_rate']
            self.banks = sett['banks']

            self.acc1 = sett['acc1']
            self.acc2 = sett['acc2']
            self.accs_rate = sett['accs_rate']
            self.accs = sett['accs']
            
            self.formats_rate = sett['formats_rate']
            self.formats = sett['formats']

            self.currs_rate = sett['currs_rate']
            self.currs = sett['currs']

            self.chances = sett['chances']
            self.sections = sett['sections']
            self.is_laundering = is_laund

    def random_time(self, start, end):
        start = timedelta(hours=start, minutes=0, seconds=0)
        end = timedelta(hours=end, minutes=59, seconds=59)

        random_seconds=random.randint(start.total_seconds(),end.total_seconds())

        random_time=timedelta(seconds=random_seconds)

        return str(random_time)

    def gen_dataframe(self, n):
        array = self.gen_rows(n)
        if self.is_laundering:
            column_values = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']
        else:
            column_values = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format']
        index_values = list(range(0, n))
        fake_df = pd.DataFrame(data = array,  
                    index = index_values,  
                    columns = column_values) 
        return fake_df


    def gen_rows(self, n):
        rows = []
        for i in range(n):
            rows.append(self.gen_row())
        return rows

    def gen_row(self):
        amount_received = self.amount()
        timestamp = self.gen_time()

        banks = self.gen_bank()
        fr_bank = banks[0]
        to_bank = banks[1]

        accs = self.gen_account()
        acc1 = accs[0]
        acc2 = accs[1]
        
        receive_curr = self.currency()
        pay_format = self.format()
        amount_paid = amount_received
        pay_curr = receive_curr

        if self.is_laundering:
            return [timestamp, fr_bank, acc1, to_bank, acc2, amount_received, receive_curr, amount_paid, pay_curr, pay_format, self.is_laundering]
        else:
            return [timestamp, fr_bank, acc1, to_bank, acc2, amount_received, receive_curr, amount_paid, pay_curr, pay_format]

    def gen_time(self):
        time = self.random_time(0, 12)
        time = '2022/09/01 ' + time[:len(time)-3]
        return time
    
    def gen_bank(self):
        if random.uniform(0, 1) < 0.9:
            bank = self.banks.sample(n=1, weights=self.banks['prob'])
        else:
            return (random.choice(self.from_banks), random.choice(self.to_banks))
        return bank['banks'].values[0]

    def gen_account(self):
        if random.uniform(0, 1) < 0.9:
            acc = self.accs.sample(n=1, weights=self.accs['prob'])
        else:
            return (random.choice(self.acc1), random.choice(self.acc2))
        return acc['accs'].values[0]

    def amount(self):
        section = random.choices(self.sections, weights=self.chances, k=1)
        return random.uniform(section[0][0], section[0][1])

    def currency(self):
        currency = self.currs.sample(n=1, weights=self.banks['prob'])
        return currency['accs'].values[0][0]

    def format(self):
        format = self.formats.sample(n=1, weights=self.banks['prob'])
        return format['accs'].values[0][0]

    def corr_matrix(self, df):
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        df.loc[:, cats] = OrdinalEncoder().fit_transform(df[cats])

        corr_matrix = round(df.corr(), 2)
        return corr_matrix

