import pandas as pd
path = r"data\crop\labels.csv"
test_path = r"data\crop\test.csv"
train_path = r"data\crop\train.csv"
valid_path = r"data\crop\valid.csv"

df = pd.read_csv(path)
df = df.sample(frac = 1)

test_frac = 0.1
valid_frac = 0.1

length = len(df)
f_valid = open(valid_path, 'w')
f_train = open(train_path, 'w')
f_test = open(test_path, 'w')

f_valid.seek(0)
f_train.seek(0)
f_test.seek(0)

f_valid.write("Name,DOF\n")
f_train.write("Name,DOF\n")
f_test.write("Name,DOF\n")

for i in range(length):
    row = df.iloc[i, :]
    if i < length*test_frac:
        f_test.write(row["Name"] +"," + str(row["DOF"]) + "\n")
    elif i < length*(test_frac + valid_frac):
        f_valid.write(row["Name"] +"," + str(row["DOF"]) + "\n")
    else:
        f_train.write(row["Name"] +"," + str(row["DOF"]) + "\n")

f_valid.close()
f_train.close()
f_test.close()