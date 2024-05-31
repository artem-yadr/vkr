import pandas as pd
import random
from datetime import timedelta
import pickle as pkl
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class DGenerator:
    def __init__(self, df=None, is_laund=False):
        if df != None:
            self.banks = df['From Bank'].unique()
            self.accs = df['Account'].unique()
            self.formats = df['Payment Format'].unique()
            self.currs = df['Receiving Currency'].unique()
            self.is_laundering = is_laund
        else:
            with open("src/pickles/dgen_settings.pkl", "rb") as f:
                sett = pkl.load(f)
            
            self.banks = sett["banks"]
            self.accs = sett["accs"]
            self.formats = sett["formats"] 
            self.currs = sett["currs"]
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
        amount_received = self.amount(0, 300000)
        timestamp = self.gen_time()
        fr_bank = self.gen_bank(self.banks)
        to_bank = self.gen_bank(self.banks)
        acc1 = self.account(self.accs)
        acc2 = self.account(self.accs)
        receive_curr = self.currency(self.currs)
        pay_format = self.format(self.formats)
        amount_paid = amount_received
        pay_curr = receive_curr

        if self.is_laundering:
            return [timestamp, fr_bank, acc1, to_bank, acc2, amount_received, receive_curr, amount_paid, pay_curr, pay_format, 1]
        else:
            return [timestamp, fr_bank, acc1, to_bank, acc2, amount_received, receive_curr, amount_paid, pay_curr, pay_format]

    def gen_time(self):
        time = self.random_time(0, 12)
        time = '2022/09/01 ' + time[:len(time)-3]
        return time

    def gen_bank(self, banks):
        bank = random.choice(banks)
        return bank
        
    def account(self, accs):
        account = random.choice(accs)
        return account

    def amount(self, min_amount, max_amount):
        return random.uniform(min_amount, max_amount)

    def currency(self, currs):
        currency = random.choice(currs)
        return currency

    def format(self, formats):
        format = random.choice(formats)
        return format
    
    def corr_matrix(self, df):
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        df.loc[:, cats] = OrdinalEncoder().fit_transform(df[cats])

        corr_matrix = round(df.corr(), 2)
        return corr_matrix

