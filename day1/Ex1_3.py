import matplotlib.pyplot as plt
import lxmls.readers.simple_data_set as sds
# import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.mira as mirac

sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1],
                       balance=0.5, split=[0.5, 0, 0.5])

# sd = srs.SentimentCorpus("books")

mira = mirac.Mira()

mira.regularizer = 1.0  # This is lambda

params_mira_sd = mira.train(sd.train_X, sd.train_y)

y_pred_train = mira.test(sd.train_X, params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)

y_pred_test = mira.test(sd.test_X, params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)

print "Mira SimpleDataset Accuracy train: %f test: %f" % (acc_train, acc_test)
# print "Mira Amazon Accuracy train: %f test: %f" % (acc_train, acc_test)

fig, axis = sd.plot_data()
fig, axis = sd.add_line(fig, axis, params_mira_sd, "Mira", "green")
plt.show()

# On simple dataset
# train: 0.900000 test: 0.940000

# On Amazon dataset
# Mira Amazon Accuracy train: 0.500000 test: 0.500000

