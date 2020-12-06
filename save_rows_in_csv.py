#%%
import csv

a1 = [1,2,3,4]
a2 = [1,1]
a3 = []
a4 = [1,1,1,1,3,3,4,5,6,7]
with open('test.csv', mode='w') as csv_file:
    #fieldnames = ['emp_name', 'dept', 'birth_month']
    #writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    #writer.writeheader()
    writer = csv.writer(csv_file)
    writer.writerow(a1)
    writer.writerow(a2)
    writer.writerow(a3)
    writer.writerow(a4)
# %%
